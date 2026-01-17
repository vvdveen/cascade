"""
ISS (Instruction Set Simulator) runner interface.

Interfaces with Spike (riscv-isa-sim) to:
1. Run intermediate programs to completion
2. Collect register values at marked program points
3. Return structured feedback for ultimate program construction
"""

import subprocess
import tempfile
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import FuzzerConfig
from ..generator.intermediate import IntermediateProgram
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
            cmd = self._build_command(elf_path, collect_feedback)

            try:
                # Run Spike
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.iss_timeout / 1000.0  # Convert to seconds
                )

                result.raw_output = proc.stdout + proc.stderr
                result.exit_code = proc.returncode

                # Parse output
                if collect_feedback:
                    result.feedback = self._parse_feedback(result.raw_output, program)

                result.success = proc.returncode == 0
                result.instruction_count = self._count_instructions(result.raw_output)

            except subprocess.TimeoutExpired:
                result.timeout = True
                result.error_message = "ISS simulation timed out"

            except FileNotFoundError:
                result.error_message = f"Spike not found at {self.spike_path}"

            except Exception as e:
                result.error_message = str(e)

        return result

    def _build_command(self, elf_path: Path, trace: bool = False) -> List[str]:
        """Build Spike command line."""
        cmd = [str(self.spike_path)]

        # ISA specification
        cmd.extend(["--isa", self.isa_string])

        # Memory configuration
        mem_start = self.config.memory.code_start
        mem_size = 0x100000  # 1MB
        cmd.extend(["-m", f"0x{mem_start:x}:0x{mem_size:x}"])

        # Instruction limit
        cmd.extend(["-l", str(self.config.iss_timeout)])

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
