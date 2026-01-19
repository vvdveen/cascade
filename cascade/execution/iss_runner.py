"""
ISS (Instruction Set Simulator) runner interface.

Interfaces with Spike (riscv-isa-sim) to:
1. Run intermediate programs to completion
2. Collect register values at marked program points
3. Return structured feedback for ultimate program construction
"""

import os
import pty
import select
import subprocess
import time
import tempfile
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import FuzzerConfig
from ..generator.intermediate import IntermediateProgram
from ..generator.ultimate import UltimateProgram
from ..generator.ultimate import ISSFeedback
from .elf_writer import ELFWriter


@dataclass
class ISSResult:
    """Result from ISS simulation."""
    success: bool = False
    timeout: bool = False
    error_message: str = ""

    # Number of instructions executed
    instruction_count: int = 0

    # Exit code (if applicable)
    exit_code: int = 0

    # Feedback for ultimate program construction
    feedback: Optional[ISSFeedback] = None

    # Raw output from ISS
    raw_output: str = ""


class ISSRunner:
    """
    Runs programs on the Spike RISC-V ISS.

    Spike provides instruction-accurate simulation and supports
    tracing register values for feedback collection.
    """

    def __init__(self, config: FuzzerConfig):
        """Initialize ISS runner."""
        self.config = config
        self.spike_path = config.spike_path
        self.elf_writer = ELFWriter(config.cpu.xlen)

        # ISA string for Spike
        from ..isa.extensions import get_isa_string
        self.isa_string = get_isa_string(config.cpu.extensions, config.cpu.xlen)

    def run(self, program: IntermediateProgram,
            collect_feedback: bool = True) -> ISSResult:
        """
        Run program on ISS.

        Args:
            program: Intermediate program to run
            collect_feedback: Whether to collect register values

        Returns:
            ISSResult with execution results and feedback
        """
        result = ISSResult()

        # Write program to temporary ELF file
        with tempfile.TemporaryDirectory() as tmpdir:
            elf_path = Path(tmpdir) / "program.elf"
            self.elf_writer.write(program, elf_path)

            # Build Spike command
            trace = collect_feedback
            if self.config.cpu.name == "kronos" and not collect_feedback:
                trace = True
            cmd = self._build_command(elf_path, trace)

            try:
                if collect_feedback or (self.config.cpu.name == "kronos" and not collect_feedback):
                    self._run_spike_with_early_exit(cmd, result)
                    if result.success and collect_feedback:
                        result.feedback = self._parse_feedback(result.raw_output, program)
                else:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.config.iss_timeout / 1000.0  # Convert to seconds
                    )

                    result.raw_output = proc.stdout + proc.stderr
                    result.exit_code = proc.returncode

                    # Check for errors
                    if proc.returncode != 0:
                        if self._is_expected_termination(result.raw_output):
                            result.success = True
                        else:
                            result.error_message = result.raw_output[:500] if result.raw_output else f"Exit code {proc.returncode}"
                            result.success = False
                    else:
                        result.success = True
                    result.instruction_count = self._count_instructions(result.raw_output)

            except subprocess.TimeoutExpired as e:
                result.timeout = True
                result.error_message = "ISS simulation timed out"
                stdout = e.stdout or ""
                stderr = e.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                result.raw_output = stdout + stderr
                if not collect_feedback:
                    result.success = True

            except FileNotFoundError:
                result.error_message = f"Spike not found at {self.spike_path}"

            except Exception as e:
                result.error_message = str(e)

        return result

    def run_trace(self, program) -> Tuple[bool, List[int], str]:
        """Run program on ISS and return PC trace."""
        pcs: List[int] = []
        raw_output = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            elf_path = Path(tmpdir) / "program.elf"
            self.elf_writer.write(program, elf_path)
            cmd = self._build_command(elf_path, trace=True)

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.iss_timeout / 1000.0
                )
                raw_output = proc.stdout + proc.stderr
                pcs = self._parse_trace_pcs(raw_output)
                if proc.returncode == 0 or self._is_expected_termination(raw_output):
                    return True, pcs, raw_output
                return False, pcs, raw_output
            except subprocess.TimeoutExpired as e:
                stdout = e.stdout or ""
                stderr = e.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                raw_output = stdout + stderr
                pcs = self._parse_trace_pcs(raw_output)
                return False, pcs, raw_output
            except Exception as e:
                return False, pcs, str(e)

    def _parse_trace_pcs(self, output: str) -> List[int]:
        """Parse PC trace from Spike output."""
        pcs: List[int] = []
        pc_pattern = re.compile(r'core\s+\d+:\s+0x([0-9a-fA-F]+)\s+\(0x[0-9a-fA-F]+\)')
        for line in output.split('\n'):
            pc_match = pc_pattern.search(line)
            if pc_match:
                pcs.append(int(pc_match.group(1), 16))
        return pcs

    def _is_expected_termination(self, output: str) -> bool:
        """Check for clean program termination via ebreak instruction.

        Only ebreak/breakpoint indicate intentional program termination.
        Exception traps (trap_instruction_access_fault, etc.) are errors,
        not successful termination.
        """
        if not output:
            return False
        lowered = output.lower()
        return "ebreak" in lowered or "breakpoint" in lowered

    def _run_spike_with_early_exit(self, cmd: List[str], result: ISSResult) -> None:
        """Run Spike and stop once termination is observed in the log."""
        timeout_seconds = self.config.iss_timeout / 1000.0
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
        )
        os.close(slave_fd)

        output_chunks: List[str] = []
        deadline = time.time() + timeout_seconds
        saw_termination = False

        while True:
            if proc.poll() is not None:
                break
            if time.time() >= deadline:
                result.timeout = True
                result.error_message = "ISS simulation timed out"
                break

            rlist, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in rlist:
                data = os.read(master_fd, 4096)
                if data:
                    text = data.decode(errors="replace")
                    output_chunks.append(text)
                    if self._is_expected_termination(text):
                        saw_termination = True
                        proc.terminate()
                        break
            if saw_termination:
                break

        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1)

        try:
            while True:
                data = os.read(master_fd, 4096)
                if not data:
                    break
                output_chunks.append(data.decode(errors="replace"))
        except OSError:
            pass
        os.close(master_fd)

        result.raw_output = "".join(output_chunks)
        result.exit_code = proc.returncode or 0
        result.instruction_count = self._count_instructions(result.raw_output)

        if result.timeout:
            result.success = False
            return

        if proc.returncode == 0 or saw_termination or self._is_expected_termination(result.raw_output):
            result.success = True
        else:
            result.success = False
            result.error_message = result.raw_output[:500] if result.raw_output else f"Exit code {proc.returncode}"

    def _build_command(self, elf_path: Path, trace: bool = False) -> List[str]:
        """Build Spike command line."""
        cmd = [str(self.spike_path)]

        # ISA specification
        cmd.extend(["--isa", self.isa_string])

        # Memory configuration - format is -m<base>:<size> (no space after -m)
        mem_start = self.config.memory.code_start
        mem_size = 0x100000  # 1MB
        cmd.append(f"-m{mem_start:#x}:{mem_size:#x}")

        # Enable logging for feedback collection
        if trace:
            cmd.append("-l")
            cmd.append("--log-commits")

        # ELF file
        cmd.append(str(elf_path))

        return cmd

    def _parse_feedback(self, output: str, program: IntermediateProgram) -> ISSFeedback:
        """
        Parse Spike output to extract register values.

        Spike's commit log format (with --log-commits):
        core   0: 0x80000000 (0x00000297) auipc t0, 0x0
        core   0: x5 0x00000000 -> 0x80000000
        """
        feedback = ISSFeedback()

        current_pc = None
        current_regs: Dict[int, int] = {}

        # Register value pattern: core N: xREG OLD -> NEW
        reg_pattern = re.compile(r'core\s+\d+:\s+x(\d+)\s+0x[0-9a-fA-F]+\s+->\s+0x([0-9a-fA-F]+)')

        # PC pattern: core N: 0xADDR (0xINSTR)
        pc_pattern = re.compile(r'core\s+\d+:\s+0x([0-9a-fA-F]+)\s+\(0x[0-9a-fA-F]+\)')

        for line in output.split('\n'):
            # Check for PC
            pc_match = pc_pattern.search(line)
            if pc_match:
                # Save previous state
                if current_pc is not None and current_regs:
                    feedback.register_values[current_pc] = dict(current_regs)
                    feedback.trace.append(current_pc)

                current_pc = int(pc_match.group(1), 16)

            # Check for register update
            reg_match = reg_pattern.search(line)
            if reg_match:
                reg_num = int(reg_match.group(1))
                reg_value = int(reg_match.group(2), 16)
                current_regs[reg_num] = reg_value

        # Save final state
        if current_pc is not None and current_regs:
            feedback.register_values[current_pc] = dict(current_regs)
            feedback.trace.append(current_pc)
            feedback.final_registers = dict(current_regs)

        return feedback

    def _count_instructions(self, output: str) -> int:
        """Count instructions executed from Spike output."""
        # Look for instruction count in output
        match = re.search(r'(\d+)\s+instructions', output)
        if match:
            return int(match.group(1))

        # Count PC lines as fallback
        return len(re.findall(r'core\s+\d+:\s+0x[0-9a-fA-F]+', output))

    def check_spike_available(self) -> Tuple[bool, str]:
        """
        Check if Spike is available.

        Returns:
            (available, message)
        """
        try:
            result = subprocess.run(
                [str(self.spike_path), "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return True, "Spike available"
        except FileNotFoundError:
            return False, f"Spike not found at {self.spike_path}"
        except subprocess.TimeoutExpired:
            return False, "Spike check timed out"
        except Exception as e:
            return False, str(e)


class MockISSRunner:
    """
    Mock ISS runner for testing without Spike.

    Simulates program execution for basic testing.
    """

    def __init__(self, config: FuzzerConfig):
        """Initialize mock ISS runner."""
        self.config = config

    def run(self, program: IntermediateProgram,
            collect_feedback: bool = True) -> ISSResult:
        """Run mock simulation."""
        result = ISSResult()
        result.success = True
        result.instruction_count = sum(b.num_instructions for b in program.blocks)

        if collect_feedback:
            result.feedback = self._generate_mock_feedback(program)

        return result

    def _generate_mock_feedback(self, program: IntermediateProgram) -> ISSFeedback:
        """Generate mock feedback with random register values."""
        import random

        feedback = ISSFeedback()

        # Generate register values at each cf-ambiguous point
        for marker in program.cf_markers:
            regs = {}
            for i in range(32):
                regs[i] = random.randint(0, 0xFFFFFFFF)
            regs[0] = 0  # x0 is always 0
            feedback.register_values[marker.pc] = regs

        # Final register state
        final = {}
        for i in range(32):
            import random
            final[i] = random.randint(0, 0xFFFFFFFF)
        final[0] = 0
        feedback.final_registers = final

        return feedback
