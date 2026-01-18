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
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import (
    FuzzerConfig, CPUConfig, MemoryLayout, InstructionWeights,
    Extension, PICORV32_CONFIG, VEXRISCV_CONFIG, ROCKET_CONFIG
)
from .generator.intermediate import IntermediateProgram, IntermediateProgramGenerator
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
            available, msg = self.rtl_runner.check_verilator_available()
            if not available:
                logger.warning(f"Verilator not available: {msg}")
                logger.warning("Using mock RTL runner")
                self.rtl_runner = MockRTLRunner(self.config)
        else:
            logger.warning("RTL model path not configured, using mock runner")
            self.rtl_runner = MockRTLRunner(self.config)

    def calibrate(self) -> bool:
        """
        Calibrate fuzzer for target CPU.

        Determines available CSRs, extension support, etc.
        """
        logger.info(f"Calibrating for {self.config.cpu.name}")

        # Generate a simple test program
        test_program = self.intermediate_gen.generate(seed=0)

        # Run on ISS to verify setup
        result = self.iss_runner.run(test_program, collect_feedback=False)

        if not result.success and not isinstance(self.iss_runner, MockISSRunner):
            logger.error(f"Calibration failed: {result.error_message}")
            logger.debug(f"ISS raw output: {result.raw_output[:1000] if result.raw_output else 'none'}")
            return False

        logger.info("Calibration successful")
        return True

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
            last_print = time.time()
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

                now = time.time()
                if now - last_print >= 0.2 or completed == num_programs:
                    self._print_progress(completed, executed, bugs_found, num_programs, worker_done, worker_totals)
                    last_print = now

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

        # 2. Run on ISS, collect feedback
        iss_result = self.iss_runner.run(intermediate, collect_feedback=True)

        if not iss_result.success:
            self.stats.iss_errors += 1
            if iss_result.timeout:
                logger.debug(f"ISS timeout at iteration {iteration}")
            return

        if iss_result.feedback is None:
            logger.warning(f"No ISS feedback at iteration {iteration}")
            return

        # 3. Generate ultimate program
        ultimate = self.ultimate_gen.generate(intermediate, iss_result.feedback)

        # 4. Run on RTL
        rtl_result = self.rtl_runner.run(ultimate)
        self.stats.programs_executed += 1

        # 5. Check for bugs
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

    def _report_bug_post(self, bug_dir: Path, reduction) -> None:
        """Save reduced artifacts and append reduction metadata."""
        if reduction and reduction.reduced_program:
            reduced_path = bug_dir / "reduced.elf"
            reduced_asm_path = bug_dir / "reduced.S"
            self.elf_writer.write(reduction.reduced_program, reduced_path)
            with open(reduced_asm_path, "w") as asm_file:
                asm_file.write(self._format_program_asm(reduction.reduced_program))

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
        if not vcd_path.exists():
            return
        pcs = self._extract_pc_trace(vcd_path, max_entries=None)
        if not pcs:
            return
        meta_path = bug_dir / "metadata.txt"
        with open(meta_path, "a") as f:
            f.write("PC trace (all):\n")
            f.write(", ".join(f"0x{pc:08x}" for pc in pcs) + "\n")

    def _extract_pc_trace(self, vcd_path: Path, max_entries: Optional[int] = 64) -> List[int]:
        """Extract PC samples from VCD (rvfi_pc_wdata) if available."""
        signal_id = None
        in_header = True

        def parse_value(line: str) -> Optional[int]:
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
                return int(bits, 2)
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

        pcs: List[int] = []
        with vcd_path.open("r", errors="replace") as vcd_file:
            for line in vcd_file:
                if in_header:
                    if line.startswith("$var"):
                        parts = line.split()
                        if len(parts) >= 5:
                            ident = parts[3]
                            name = parts[4]
                            if name.endswith("rvfi_pc_wdata"):
                                signal_id = ident
                    elif line.startswith("$enddefinitions"):
                        in_header = False
                    continue

                if signal_id is None:
                    break

                value = parse_value(line)
                if value is None:
                    continue
                pcs.append(value)
                if max_entries is not None and len(pcs) > max_entries:
                    pcs = pcs[-max_entries:]

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
        worker_parts = []
        for worker_id in sorted(per_worker_total.keys()):
            done = per_worker_done.get(worker_id, 0)
            total_worker = per_worker_total[worker_id]
            worker_parts.append(f"W{worker_id} { _format_bar(done, total_worker, width=10) }")
        line = f"\rOverall {overall} done {completed}/{total} bugs {bugs_found}  " + "  ".join(worker_parts)
        print(line, end="", flush=True)


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
    fuzzer = Fuzzer(config)
    fuzzer.stats = FuzzerStats()
    fuzzer.bugs = []

    for i in range(count):
        iteration = start_index + i
        try:
            prev_executed = fuzzer.stats.programs_executed
            prev_bugs = fuzzer.stats.bugs_found
            fuzzer._fuzz_iteration(iteration)
        except Exception as e:
            logger.error(f"Worker {worker_id} error at iteration {iteration}: {e}")
        executed_delta = fuzzer.stats.programs_executed - prev_executed
        bug_delta = fuzzer.stats.bugs_found - prev_bugs
        progress_queue.put((worker_id, executed_delta, bug_delta))

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
        choices=["picorv32", "vexriscv", "rocket", "custom"],
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
    elif args.cpu == "vexriscv":
        cpu_config = VEXRISCV_CONFIG
    elif args.cpu == "rocket":
        cpu_config = ROCKET_CONFIG
    else:
        cpu_config = CPUConfig(name="custom")

    # Create fuzzer configuration
    config = FuzzerConfig(
        cpu=cpu_config,
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
