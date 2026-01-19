"""
Main Cascade fuzzer implementation.

Orchestrates program generation, ISS pre-simulation, RTL execution,
and bug detection.
"""

import argparse
import logging
import sys
import time
import multiprocessing
import queue
import os
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import (
    FuzzerConfig, CPUConfig, MemoryLayout, InstructionWeights,
    Extension, PICORV32_CONFIG, KRONOS_CONFIG, KRONOS_MEMORY_LAYOUT,
    VEXRISCV_CONFIG, ROCKET_CONFIG
)
from .generator.intermediate import IntermediateProgram, IntermediateProgramGenerator
from .generator.basic_block import BasicBlock
from .isa.encoding import EncodedInstruction
from .isa.instructions import ADDI, EBREAK, LUI, SW
from .generator.ultimate import UltimateProgram, UltimateProgramGenerator
from .execution.iss_runner import ISSRunner, MockISSRunner
from .execution.rtl_runner import RTLRunner, MockRTLRunner
from .execution.elf_writer import ELFWriter
from .reduction.reducer import Reducer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cascade')

class SeedCollisionError(RuntimeError):
    """Raised when a program seed has already been used."""


@dataclass
class BugReport:
    """Report for a detected bug."""
    bug_id: str
    timestamp: datetime
    program_seed: int
    program_path: Path
    reduced_program_path: Optional[Path] = None
    description: str = ""
    cycle_count: int = 0
    instruction_count: int = 0


@dataclass
class FuzzerStats:
    """Statistics from fuzzing run."""
    programs_generated: int = 0
    programs_executed: int = 0
    bugs_found: int = 0
    iss_errors: int = 0
    rtl_errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if self.start_time is None:
            return 0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def programs_per_second(self) -> float:
        """Programs generated per second."""
        if self.duration == 0:
            return 0
        return self.programs_generated / self.duration


class Fuzzer:
    """
    Main Cascade fuzzer.

    Implements the fuzzing loop:
    1. Generate intermediate program
    2. Run on ISS, collect feedback
    3. Generate ultimate program
    4. Run on RTL, detect bugs
    5. Reduce bug-triggering programs
    """

    def __init__(self, config: FuzzerConfig):
        """Initialize fuzzer."""
        self.config = config
        self.stats = FuzzerStats()
        self.bugs: List[BugReport] = []

        # Initialize generators
        self.intermediate_gen = IntermediateProgramGenerator(config)
        self.ultimate_gen = UltimateProgramGenerator(config)

        # Initialize runners
        self._init_runners()

        # ELF writer
        self.elf_writer = ELFWriter(config.cpu.xlen)
        self.reducer = Reducer(config, self.rtl_runner, self.iss_runner)

        # Output directory
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._seed_history_path = self.output_dir / "seed_history.txt"

    def _init_runners(self) -> None:
        """Initialize ISS and RTL runners."""
        # Check if Spike is available
        self.iss_runner = ISSRunner(self.config)
        available, msg = self.iss_runner.check_spike_available()
        if not available:
            logger.warning(f"Spike not available: {msg}")
            logger.warning("Using mock ISS runner")
            self.iss_runner = MockISSRunner(self.config)

        # Check if RTL simulation is available
        if self.config.rtl_model_path:
            self.rtl_runner = RTLRunner(self.config)
            # Check if pre-built simulation binary exists
            sim_binary = self.rtl_runner._get_sim_binary()
            if sim_binary is None:
                # No pre-built binary, check if we can build with Verilator
                available, msg = self.rtl_runner.check_verilator_available()
                if not available:
                    logger.warning(f"No simulation binary found and Verilator not available: {msg}")
                    logger.warning("Using mock RTL runner")
                    self.rtl_runner = MockRTLRunner(self.config)
            else:
                logger.info(f"Using RTL simulation binary: {sim_binary}")
        else:
            logger.warning("RTL model path not configured, using mock runner")
            self.rtl_runner = MockRTLRunner(self.config)

    def calibrate(self) -> bool:
        """
        Calibrate fuzzer for target CPU.

        Determines available CSRs, extension support, etc.
        """
        logger.info(f"Calibrating for {self.config.cpu.name}")

        # Use a deterministic, safe program to avoid ISS timeouts.
        test_program = self._build_calibration_program()

        # Run on ISS to verify setup
        result = self.iss_runner.run(test_program, collect_feedback=False)

        if not result.success and not isinstance(self.iss_runner, MockISSRunner):
            logger.error(f"Calibration failed: {result.error_message}")
            logger.debug(f"ISS raw output: {result.raw_output[:1000] if result.raw_output else 'none'}")
            return False

        logger.info("Calibration successful")
        return True

    def _build_calibration_program(self) -> IntermediateProgram:
        """Build a minimal program that should terminate on the ISS."""
        code_start = self.config.memory.code_start
        data_start = self.config.memory.data_start
        block = BasicBlock(start_addr=code_start, block_id=0)

        # Minimal setup so the program executes a real instruction before exiting.
        block.instructions.append(EncodedInstruction(ADDI, rd=1, rs1=0, imm=0))

        if self.config.cpu.name == "kronos":
            # Write to tohost for kronos compliance semantics.
            upper = (data_start + 0x800) >> 12
            lower = data_start - (upper << 12)
            block.instructions.append(EncodedInstruction(LUI, rd=2, imm=(upper & 0xFFFFF) << 12))
            block.instructions.append(EncodedInstruction(ADDI, rd=2, rs1=2, imm=lower & 0xFFF))
            block.instructions.append(EncodedInstruction(ADDI, rd=1, rs1=0, imm=1))
            block.instructions.append(EncodedInstruction(SW, rs1=2, rs2=1, imm=0))

        block.terminator = EncodedInstruction(EBREAK)

        program = IntermediateProgram(
            blocks=[block],
            entry_addr=code_start,
            code_start=code_start,
            code_end=block.end_addr,
            data_start=data_start,
            data_end=data_start + self.config.memory.data_size,
        )
        return program

    def fuzz(self, num_programs: Optional[int] = None) -> List[BugReport]:
        """
        Run the main fuzzing loop.

        Args:
            num_programs: Number of programs to generate (default from config)

        Returns:
            List of bug reports
        """
        if num_programs is None:
            num_programs = self.config.num_programs

        logger.info(f"Starting fuzzing run: {num_programs} programs")

        self.stats = FuzzerStats()
        self.stats.start_time = datetime.now()
        self.bugs = []

        if self.config.num_workers > 1:
            self._fuzz_parallel(num_programs)
        else:
            self._fuzz_serial(num_programs)

        self.stats.end_time = datetime.now()
        self._log_final_stats()

        return self.bugs

    def _fuzz_serial(self, num_programs: int) -> None:
        """Run fuzzing serially with progress updates."""
        last_print = time.time()
        completed = 0
        executed = 0
        for i in range(num_programs):
            try:
                prev_executed = self.stats.programs_executed
                prev_bugs = self.stats.bugs_found
                self._fuzz_iteration(i)
                executed += self.stats.programs_executed - prev_executed
            except SeedCollisionError as e:
                logger.warning(str(e))
                raise SystemExit(1)
            except Exception as e:
                logger.error(f"Error in iteration {i}: {e}")
            completed += 1

            now = time.time()
            if now - last_print >= 0.2 or (i + 1) == num_programs:
                self._print_progress(
                    completed,
                    executed,
                    self.stats.bugs_found,
                    num_programs,
                    {0: i + 1},
                    {0: num_programs},
                )
                last_print = now
        print()

    def _fuzz_parallel(self, num_programs: int) -> None:
        """Run fuzzing with multiple workers."""
        ctx = multiprocessing.get_context("spawn")
        manager = multiprocessing.Manager()
        progress_queue: multiprocessing.Queue = manager.Queue()

        worker_counts = _split_work(num_programs, self.config.num_workers)
        worker_totals = {worker_id: count for worker_id, count in enumerate(worker_counts)}
        worker_done = {worker_id: 0 for worker_id in worker_totals}

        futures = []
        start_index = 0
        with ProcessPoolExecutor(max_workers=self.config.num_workers, mp_context=ctx) as executor:
            for worker_id, count in enumerate(worker_counts):
                if count == 0:
                    continue
                futures.append(
                    executor.submit(
                        _worker_fuzz,
                        self.config,
                        start_index,
                        count,
                        worker_id,
                        progress_queue,
                    )
                )
                start_index += count

            completed = 0
            executed = 0
            bugs_found = 0
            last_completed = 0
            while completed < num_programs:
                try:
                    worker_id, executed_delta, bug_delta = progress_queue.get(timeout=0.1)
                    if worker_id in worker_done:
                        worker_done[worker_id] += 1
                        completed += 1
                        executed += executed_delta
                        bugs_found += bug_delta
                except queue.Empty:
                    pass

                # Only print when progress was made
                if completed > last_completed:
                    self._print_progress(completed, executed, bugs_found, num_programs, worker_done, worker_totals)
                    last_completed = completed

                if all(f.done() for f in futures):
                    for future in futures:
                        exc = future.exception()
                        if exc:
                            raise exc
                    break

            print()

            for future in as_completed(futures):
                worker_stats, worker_bugs = future.result()
                self._merge_worker_stats(worker_stats, worker_bugs)

    def _fuzz_iteration(self, iteration: int) -> None:
        """Run a single fuzzing iteration."""
        seed = self.config.seed
        if seed is not None:
            seed = seed + iteration

        # 1. Generate intermediate program
        intermediate = self.intermediate_gen.generate(seed=seed)
        self.stats.programs_generated += 1
        program_seed = intermediate.descriptor.seed if intermediate.descriptor else seed
        if program_seed is not None:
            self._record_seed(program_seed)

        # 2. Run on ISS, collect feedback
        iss_result = self.iss_runner.run(intermediate, collect_feedback=True)

        if not iss_result.success:
            self.stats.iss_errors += 1
            logger.warning(f"ISS failed at iteration {iteration}: {iss_result.error_message}")
            if iss_result.timeout:
                try:
                    self._report_iss_timeout(
                        iteration,
                        intermediate,
                        intermediate,
                        iss_result,
                        label="intermediate"
                    )
                except Exception as e:
                    logger.error(f"Failed to save ISS timeout report at iteration {iteration}: {e}")
            return

        if iss_result.feedback is None:
            logger.debug(f"No ISS feedback at iteration {iteration}")
            return

        # 3. Generate ultimate program
        ultimate = self.ultimate_gen.generate(intermediate, iss_result.feedback)

        # 4. Verify ultimate program terminates on ISS before RTL
        ultimate_iss = self.iss_runner.run(ultimate, collect_feedback=False)
        if not ultimate_iss.success:
            self.stats.iss_errors += 1
            if ultimate_iss.timeout:
                logger.debug(f"Ultimate ISS timeout at iteration {iteration}")
                try:
                    self._report_iss_timeout(
                        iteration,
                        intermediate,
                        ultimate,
                        ultimate_iss
                    )
                except Exception as e:
                    logger.error(f"Failed to save ISS timeout report at iteration {iteration}: {e}")
            return

        # 5. Run on RTL
        rtl_result = self.rtl_runner.run(ultimate)
        self.stats.programs_executed += 1

        # 6. Check for bugs
        if rtl_result.bug_detected:
            self.stats.bugs_found += 1
            bug, bug_dir = self._report_bug_pre(
                iteration, intermediate, ultimate, rtl_result
            )
            if rtl_result.timeout:
                try:
                    trace_dir = bug_dir / "rtl_trace"
                    self.rtl_runner.capture_trace(ultimate, trace_dir)
                    self._append_trace_summary(bug_dir, trace_dir)
                except Exception as e:
                    logger.error(f"Failed to capture RTL trace at iteration {iteration}: {e}")
            reduction = None
            try:
                reduction = self.reducer.reduce(ultimate, iss_result.feedback)
            except Exception as e:
                logger.error(f"Reduction failed at iteration {iteration}: {e}")
            try:
                self._report_bug_post(bug_dir, reduction)
                if reduction and reduction.reduced_program:
                    bug.reduced_program_path = bug_dir / "reduced.elf"
                self.bugs.append(bug)
                logger.info(f"Bug detected! ID: {bug.bug_id}")
            except Exception as e:
                logger.error(f"Failed to save bug artifacts at iteration {iteration}: {e}")

        if not rtl_result.success and not rtl_result.bug_detected:
            self.stats.rtl_errors += 1

    def _report_bug_pre(self, iteration: int,
                        intermediate: IntermediateProgram,
                        ultimate: UltimateProgram,
                        rtl_result) -> Tuple[BugReport, Path]:
        """Save pre-reduction bug artifacts and metadata."""
        timestamp = datetime.now()
        bug_id = f"bug_{timestamp.strftime('%Y%m%d_%H%M%S')}_{iteration}"

        # Create bug directory
        bug_dir = self.output_dir / "bugs" / bug_id
        bug_dir.mkdir(parents=True, exist_ok=True)

        # Save programs
        ultimate_path = bug_dir / "ultimate.elf"
        intermediate_path = bug_dir / "intermediate.elf"
        ultimate_asm_path = bug_dir / "ultimate.S"
        rerun_script_path = bug_dir / "rerun_rtl.sh"

        self.elf_writer.write(ultimate, ultimate_path)
        self.elf_writer.write(intermediate, intermediate_path)
        with open(ultimate_asm_path, "w") as asm_file:
            asm_file.write(self._format_program_asm(ultimate))
        self._write_rerun_script(rerun_script_path, intermediate.descriptor.seed if intermediate.descriptor else 0)
        self._write_iss_trace(bug_dir / "iss_trace_ultimate.txt", ultimate)

        # Save metadata
        meta_path = bug_dir / "metadata.txt"
        with open(meta_path, 'w') as f:
            f.write(f"Bug ID: {bug_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Seed: {intermediate.descriptor.seed if intermediate.descriptor else 'unknown'}\n")
            f.write(f"Blocks: {len(ultimate.blocks)}\n")
            f.write(f"Cycle count: {rtl_result.cycle_count}\n")
            f.write(f"RTL timeout: {rtl_result.timeout}\n")
            f.write("Reduction: pending\n")
            f.write(f"RTL output:\n{rtl_result.raw_output}\n")

        bug = BugReport(
            bug_id=bug_id,
            timestamp=timestamp,
            program_seed=intermediate.descriptor.seed if intermediate.descriptor else 0,
            program_path=ultimate_path,
            cycle_count=rtl_result.cycle_count,
            description=rtl_result.error_message
        )
        return bug, bug_dir

    def _report_iss_timeout(self, iteration: int,
                            intermediate: IntermediateProgram,
                            ultimate: UltimateProgram,
                            iss_result,
                            label: str = "ultimate") -> None:
        """Save ISS timeout artifacts in output/errors."""
        timestamp = datetime.now()
        report_id = f"iss_timeout_{label}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{iteration}"
        error_dir = self.output_dir / "errors" / report_id
        error_dir.mkdir(parents=True, exist_ok=True)

        ultimate_path = error_dir / "ultimate.elf"
        intermediate_path = error_dir / "intermediate.elf"
        ultimate_asm_path = error_dir / "ultimate.S"
        iss_script_path = error_dir / "rerun_iss.sh"
        meta_path = error_dir / "metadata.txt"

        self.elf_writer.write(ultimate, ultimate_path)
        self.elf_writer.write(intermediate, intermediate_path)
        with open(ultimate_asm_path, "w") as asm_file:
            asm_file.write(self._format_program_asm(ultimate))
        self._write_rerun_iss_script(iss_script_path, ultimate_path)

        with open(meta_path, "w") as f:
            f.write(f"Report ID: {report_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Seed: {intermediate.descriptor.seed if intermediate.descriptor else 'unknown'}\n")
            f.write(f"Blocks: {len(ultimate.blocks)}\n")
            f.write("ISS timeout: True\n")
            f.write(f"ISS output:\n{iss_result.raw_output}\n")

    def _report_bug_post(self, bug_dir: Path, reduction) -> None:
        """Save reduced artifacts and append reduction metadata."""
        if reduction and reduction.reduced_program:
            reduced_path = bug_dir / "reduced.elf"
            reduced_asm_path = bug_dir / "reduced.S"
            self.elf_writer.write(reduction.reduced_program, reduced_path)
            with open(reduced_asm_path, "w") as asm_file:
                asm_file.write(self._format_program_asm(reduction.reduced_program))
            self._write_iss_trace(bug_dir / "iss_trace_reduced.txt", reduction.reduced_program)

        meta_path = bug_dir / "metadata.txt"
        with open(meta_path, "a") as f:
            if reduction and reduction.reduced_program:
                f.write(f"Reduced blocks: {len(reduction.reduced_program.blocks)}\n")
                f.write(f"Reduced instructions: {reduction.reduced_size}\n")
                f.write(f"Reduction ratio: {reduction.reduction_ratio:.1%}\n")
            else:
                f.write("Reduction: failed or unavailable\n")

    def _append_trace_summary(self, bug_dir: Path, trace_dir: Path) -> None:
        """Append PC trace summary from VCD to metadata."""
        vcd_path = trace_dir / "testbench.vcd"
        pcs = []
        source = "unknown"
        if vcd_path.exists():
            pcs, source = self._extract_pc_trace(vcd_path, max_entries=None)
        if not pcs:
            trace_path = trace_dir / "testbench.trace"
            if trace_path.exists():
                pcs = self._extract_pc_trace_from_tracefile(trace_path)
                source = "testbench.trace"
        if not pcs:
            meta_path = bug_dir / "metadata.txt"
            with open(meta_path, "a") as f:
                f.write("PC trace: unavailable\n")
            return
        meta_path = bug_dir / "metadata.txt"
        with open(meta_path, "a") as f:
            f.write(f"PC trace (all, source={source}):\n")
            f.write(", ".join(f"0x{pc:08x}" for pc in pcs) + "\n")
            unique_pcs = len(set(pcs))
            f.write(f"PC samples: {len(pcs)}, unique: {unique_pcs}\n")
            if unique_pcs == 1:
                resetn_high = self._vcd_signal_seen_high(vcd_path, "resetn")
                f.write(f"PC appears stuck at 0x{pcs[0]:08x}; resetn_high={resetn_high}\n")

    def _extract_pc_trace(self, vcd_path: Path,
                          max_entries: Optional[int] = 64) -> Tuple[List[int], str]:
        """Extract PC samples from VCD (rvfi_pc_wdata, reg_pc, trace_data, or mem_axi_araddr)."""
        candidates = {}
        in_header = True

        def parse_value(line: str, signal_id: str, signal_kind: str) -> Optional[int]:
            line = line.strip()
            if not line:
                return None
            if line.startswith("b"):
                parts = line[1:].split()
                if len(parts) != 2:
                    return None
                bits, ident = parts
                if ident != signal_id:
                    return None
                if any(ch in "xXzZ" for ch in bits):
                    return None
                value = int(bits, 2)
                if signal_kind == "trace":
                    return value & 0xFFFFFFFF
                return value
            if line.startswith("h"):
                parts = line[1:].split()
                if len(parts) != 2:
                    return None
                value, ident = parts
                if ident != signal_id:
                    return None
                if any(ch in "xXzZ" for ch in value):
                    return None
                return int(value, 16)
            if line[0] in "01" and len(line) > 1:
                ident = line[1:]
                if ident != signal_id:
                    return None
                return int(line[0], 2)
            return None

        def read_pcs(signal_id: str, signal_kind: str) -> List[int]:
            pcs_local: List[int] = []
            with vcd_path.open("r", errors="replace") as vcd_file:
                header_done = False
                for line in vcd_file:
                    if not header_done:
                        if line.lstrip().startswith("$enddefinitions"):
                            header_done = True
                        continue
                    value = parse_value(line, signal_id, signal_kind)
                    if value is None:
                        continue
                    pcs_local.append(value)
                    if max_entries is not None and len(pcs_local) > max_entries:
                        pcs_local = pcs_local[-max_entries:]
            return pcs_local

        source = "unknown"
        with vcd_path.open("r", errors="replace") as vcd_file:
            for line in vcd_file:
                if in_header:
                    stripped = line.lstrip()
                    if stripped.startswith("$var"):
                        parts = stripped.split()
                        if len(parts) >= 5:
                            ident = parts[3]
                            name = parts[4]
                            if name.endswith("rvfi_pc_wdata"):
                                candidates["rvfi_pc_wdata"] = ident
                            elif name.endswith("reg_pc"):
                                candidates["reg_pc"] = ident
                            elif name.endswith("mem_axi_araddr"):
                                candidates["mem_axi_araddr"] = ident
                            elif name.endswith("trace_data"):
                                candidates["trace_data"] = ident
                    elif stripped.startswith("$enddefinitions"):
                        in_header = False
                    continue
            # only need header

        priority = [
            ("rvfi_pc_wdata", "pc"),
            ("reg_pc", "pc"),
            ("mem_axi_araddr", "pc"),
            ("trace_data", "trace"),
        ]
        for name, kind in priority:
            signal_id = candidates.get(name)
            if not signal_id:
                continue
            pcs = read_pcs(signal_id, kind)
            if len(pcs) > 1:
                return pcs, name

        # Fall back to whatever we have.
        for name, kind in priority:
            signal_id = candidates.get(name)
            if not signal_id:
                continue
            pcs = read_pcs(signal_id, kind)
            if pcs:
                return pcs, name

        return [], source

    def _vcd_signal_seen_high(self, vcd_path: Path, signal_name: str) -> bool:
        """Check if a single-bit signal ever goes high in VCD."""
        signal_id = None
        in_header = True
        with vcd_path.open("r", errors="replace") as vcd_file:
            for line in vcd_file:
                if in_header:
                    stripped = line.lstrip()
                    if stripped.startswith("$var"):
                        parts = stripped.split()
                        if len(parts) >= 5 and parts[4].endswith(signal_name):
                            signal_id = parts[3]
                    elif stripped.startswith("$enddefinitions"):
                        in_header = False
                    continue
                if signal_id is None:
                    break
                line = line.strip()
                if line.startswith("1") and line[1:] == signal_id:
                    return True
        return False

    def _extract_pc_trace_from_tracefile(self, trace_path: Path) -> List[int]:
        """Extract PC samples from testbench.trace."""
        pcs: List[int] = []
        with trace_path.open("r", errors="replace") as trace_file:
            for line in trace_file:
                value = line.strip()
                if not value:
                    continue
                try:
                    raw = int(value, 16)
                except ValueError:
                    continue
                pcs.append(raw & 0xFFFFFFFF)
        return pcs

    def _format_program_asm(self, program: UltimateProgram) -> str:
        """Format ultimate program as simple assembly listing."""
        lines = []
        for block in sorted(program.blocks, key=lambda b: b.start_addr):
            lines.append(f"# block {block.block_id} @ 0x{block.start_addr:08x}")
            pc = block.start_addr
            for instr in block.instructions:
                lines.append(f"0x{pc:08x}: {instr.to_asm()}")
                pc += 4
            if block.terminator:
                lines.append(f"0x{pc:08x}: {block.terminator.to_asm()}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _write_iss_trace(self, output_path: Path, program: UltimateProgram) -> None:
        """Write ISS PC trace for a program if Spike is available."""
        if isinstance(self.iss_runner, MockISSRunner):
            return
        success, pcs, _ = self.iss_runner.run_trace(program)
        if not pcs:
            return
        with open(output_path, "w") as trace_file:
            trace_file.write("# ISS PC trace\n")
            trace_file.write(f"# success={success}\n")
            for pc in pcs:
                trace_file.write(f"0x{pc:08x}\n")

    def _write_rerun_script(self, script_path: Path, seed: int) -> None:
        """Write a helper script to re-run the RTL simulator for a bug."""
        spike_path = self.config.spike_path
        rtl_path = self.config.rtl_model_path or Path("deps/picorv32")
        script = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "\n"
            f'export PATH="{spike_path.parent}:$PATH"\n'
            f'cascade -n 1 --cpu {self.config.cpu.name} '
            f'--seed {seed} '
            f'--spike-path "{spike_path}" '
            f'--rtl-path "{rtl_path}"\n'
        )
        script_path.write_text(script)
        script_path.chmod(0o755)

    def _write_rerun_iss_script(self, script_path: Path, ultimate_path: Path) -> None:
        """Write a helper script to re-run ISS on an ultimate program."""
        spike_path = self.config.spike_path
        script = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "\n"
            f'export PATH="{spike_path.parent}:$PATH"\n'
            f'"{spike_path}" --isa {self.iss_runner.isa_string} '
            f'-m0x80000000:0x100000 -l --log-commits "{ultimate_path}"\n'
        )
        script_path.write_text(script)
        script_path.chmod(0o755)

    def _log_progress(self) -> None:
        """Log current progress."""
        logger.info(
            f"Progress: {self.stats.programs_generated} programs, "
            f"{self.stats.bugs_found} bugs, "
            f"{self.stats.programs_per_second:.1f} prog/s"
        )

    def _log_final_stats(self) -> None:
        """Log final statistics."""
        logger.info("=" * 50)
        logger.info("Fuzzing Complete")
        logger.info("=" * 50)
        logger.info(f"Programs generated: {self.stats.programs_generated}")
        logger.info(f"Programs executed: {self.stats.programs_executed}")
        logger.info(f"Bugs found: {self.stats.bugs_found}")
        logger.info(f"ISS errors: {self.stats.iss_errors}")
        logger.info(f"RTL errors: {self.stats.rtl_errors}")
        logger.info(f"Duration: {self.stats.duration:.1f}s")
        logger.info(f"Rate: {self.stats.programs_per_second:.1f} programs/second")

    def _record_seed(self, seed: int) -> None:
        """Record a seed and raise if it has been used before."""
        self._seed_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._seed_history_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            existing = set()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing.add(int(line))
                except ValueError:
                    continue
            if seed in existing:
                raise SeedCollisionError(
                    f"Seed collision detected: seed {seed} already used in {self._seed_history_path}"
                )
            f.write(f"{seed}\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

    def _merge_worker_stats(self, worker_stats: FuzzerStats, worker_bugs: List[BugReport]) -> None:
        """Merge worker stats and bug reports into this fuzzer's stats."""
        self.stats.programs_generated += worker_stats.programs_generated
        self.stats.programs_executed += worker_stats.programs_executed
        self.stats.bugs_found += worker_stats.bugs_found
        self.stats.iss_errors += worker_stats.iss_errors
        self.stats.rtl_errors += worker_stats.rtl_errors
        self.bugs.extend(worker_bugs)

    def _print_progress(self, completed: int, executed: int, bugs_found: int, total: int,
                        per_worker_done: dict, per_worker_total: dict) -> None:
        """Print progress bars for overall and per-worker runs."""
        overall = _format_bar(executed, total)
        line = f"Overall {overall} done {completed}/{total} bugs {bugs_found}"
        print(line, flush=True)


def _format_bar(done: int, total: int, width: int = 20) -> str:
    """Format a simple ASCII progress bar."""
    if total <= 0:
        return f"[{'#' * width}] 0/0"
    filled = int(width * done / total)
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total}"


def _split_work(total: int, workers: int) -> List[int]:
    """Split work across workers as evenly as possible."""
    if workers <= 0:
        return [total]
    base = total // workers
    remainder = total % workers
    return [base + (1 if i < remainder else 0) for i in range(workers)]


def _worker_fuzz(config: FuzzerConfig, start_index: int, count: int,
                 worker_id: int, progress_queue: multiprocessing.Queue) -> Tuple[FuzzerStats, List[BugReport]]:
    """Worker process entrypoint for parallel fuzzing."""
    # Set up worker-specific logging
    worker_log = config.output_dir / f"worker_{worker_id}.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(worker_log)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    worker_logger = logging.getLogger(f'cascade.worker{worker_id}')
    worker_logger.addHandler(file_handler)
    worker_logger.setLevel(logging.DEBUG)

    worker_logger.info(f"Worker {worker_id} starting, count={count}")

    try:
        fuzzer = Fuzzer(config)
    except Exception as e:
        worker_logger.error(f"Failed to create Fuzzer: {e}", exc_info=True)
        raise

    worker_logger.info(f"Worker {worker_id} fuzzer created")
    fuzzer.stats = FuzzerStats()
    fuzzer.bugs = []

    for i in range(count):
        iteration = start_index + i
        try:
            prev_executed = fuzzer.stats.programs_executed
            prev_bugs = fuzzer.stats.bugs_found
            fuzzer._fuzz_iteration(iteration)
            if i < 3:  # Log first few iterations
                worker_logger.info(f"Iteration {iteration}: generated={fuzzer.stats.programs_generated}, executed={fuzzer.stats.programs_executed}, iss_errors={fuzzer.stats.iss_errors}")
        except SeedCollisionError as e:
            worker_logger.warning(str(e))
            raise
        except Exception as e:
            worker_logger.error(f"Error at iteration {iteration}: {e}", exc_info=True)
        executed_delta = fuzzer.stats.programs_executed - prev_executed
        bug_delta = fuzzer.stats.bugs_found - prev_bugs
        progress_queue.put((worker_id, executed_delta, bug_delta))

    worker_logger.info(f"Worker {worker_id} finished")
    return fuzzer.stats, fuzzer.bugs


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Cascade: RISC-V CPU Fuzzer"
    )

    parser.add_argument(
        "-n", "--num-programs",
        type=int,
        default=100,
        help="Number of programs to generate"
    )

    parser.add_argument(
        "-c", "--cpu",
        choices=["picorv32", "kronos", "vexriscv", "rocket", "custom"],
        default="picorv32",
        help="Target CPU"
    )

    parser.add_argument(
        "--rtl-path",
        type=Path,
        help="Path to RTL model"
    )

    parser.add_argument(
        "--spike-path",
        type=Path,
        default=Path("/opt/riscv/bin/spike"),
        help="Path to Spike ISS"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./output"),
        help="Output directory"
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Only run calibration, don't fuzz"
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=(os.cpu_count() or 1),
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Select CPU configuration
    if args.cpu == "picorv32":
        cpu_config = PICORV32_CONFIG
        memory_layout = MemoryLayout()
    elif args.cpu == "kronos":
        cpu_config = KRONOS_CONFIG
        memory_layout = KRONOS_MEMORY_LAYOUT
    elif args.cpu == "vexriscv":
        cpu_config = VEXRISCV_CONFIG
        memory_layout = MemoryLayout()
    elif args.cpu == "rocket":
        cpu_config = ROCKET_CONFIG
        memory_layout = MemoryLayout()
    else:
        cpu_config = CPUConfig(name="custom")
        memory_layout = MemoryLayout()

    # Create fuzzer configuration
    config = FuzzerConfig(
        cpu=cpu_config,
        memory=memory_layout,
        num_programs=args.num_programs,
        output_dir=args.output,
        spike_path=args.spike_path,
        rtl_model_path=args.rtl_path,
        seed=args.seed,
        num_workers=args.workers,
    )

    # Create and run fuzzer
    fuzzer = Fuzzer(config)

    # Calibrate
    if not fuzzer.calibrate():
        logger.error("Calibration failed")
        sys.exit(1)

    if args.calibrate_only:
        logger.info("Calibration complete, exiting")
        sys.exit(0)

    # Fuzz
    bugs = fuzzer.fuzz()

    if bugs:
        logger.info(f"Found {len(bugs)} bugs!")
        for bug in bugs:
            logger.info(f"  - {bug.bug_id}: {bug.program_path}")

    sys.exit(0 if not bugs else 1)


if __name__ == "__main__":
    main()
