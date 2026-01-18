"""
RTL simulation runner interface.

Interfaces with Verilator to run programs on RTL CPU models.
Bug detection is based on timeout (non-termination).
"""

import subprocess
import tempfile
import shutil
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import FuzzerConfig
from ..generator.ultimate import UltimateProgram
from .elf_writer import ELFWriter, write_hex


@dataclass
class RTLResult:
    """Result from RTL simulation."""
    success: bool = False
    timeout: bool = False
    error_message: str = ""

    # Number of cycles executed
    cycle_count: int = 0

    # Bug detected (timeout = potential bug)
    bug_detected: bool = False

    # Exit status
    exit_code: int = 0

    # Register values at end (if available)
    final_registers: Dict[int, int] = field(default_factory=dict)

    # Raw output
    raw_output: str = ""


class RTLRunner:
    """
    Runs programs on RTL simulations via Verilator.

    Supports various RISC-V CPU implementations:
    - PicoRV32
    - VexRiscv
    - Rocket
    - CVA6
    - BOOM
    """

    def __init__(self, config: FuzzerConfig):
        """Initialize RTL runner."""
        self.config = config
        self.verilator_path = config.verilator_path
        self.rtl_model_path = config.rtl_model_path
        self.elf_writer = ELFWriter(config.cpu.xlen)

        # Pre-built simulation binary path
        self._sim_binary: Optional[Path] = None

    def run(self, program: UltimateProgram) -> RTLResult:
        """
        Run program on RTL simulation.

        Args:
            program: Ultimate program to run

        Returns:
            RTLResult with simulation results
        """
        result = RTLResult()

        if self.rtl_model_path is None:
            result.error_message = "RTL model path not configured"
            return result

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write program as hex file for Verilog $readmemh
            hex_path = tmpdir / "program.hex"
            write_hex(program, hex_path, base_address=0)

            # Also write ELF for reference
            elf_path = tmpdir / "program.elf"
            self.elf_writer.write(program, elf_path)

            # Run simulation
            result = self._run_simulation(hex_path, elf_path, tmpdir)

        return result

    def _run_simulation(self, hex_path: Path, elf_path: Path,
                        work_dir: Path, extra_args: Optional[List[str]] = None) -> RTLResult:
        """
        Run the RTL simulation.

        Implementation depends on the target CPU.
        """
        result = RTLResult()

        # Get simulation binary
        sim_binary = self._get_sim_binary()
        if sim_binary is None:
            result.error_message = "RTL simulation binary not available"
            return result

        # Build simulation command
        cmd = self._build_command(sim_binary, hex_path, elf_path, extra_args)

        try:
            # Run simulation with timeout
            timeout_seconds = self.config.rtl_timeout / 1000.0
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(work_dir)
            )

            result.raw_output = proc.stdout + proc.stderr
            result.exit_code = proc.returncode
            result.success = proc.returncode == 0

            # Parse cycle count and other info
            self._parse_output(result)

        except subprocess.TimeoutExpired:
            result.timeout = True
            result.bug_detected = True  # Timeout indicates potential bug
            result.error_message = "RTL simulation timed out - potential bug"

        except FileNotFoundError:
            result.error_message = f"Simulation binary not found: {sim_binary}"

        except Exception as e:
            result.error_message = str(e)

        return result

    def _get_sim_binary(self) -> Optional[Path]:
        """Get or build the simulation binary."""
        if self._sim_binary and self._sim_binary.exists():
            return self._sim_binary

        if self.rtl_model_path is None:
            return None

        # Look for pre-built binary
        possible_binaries = [
            self.rtl_model_path / "obj_dir" / "Vtestbench",
            self.rtl_model_path / "build" / "sim",
            self.rtl_model_path / "sim" / "testbench",
            self.rtl_model_path / "testbench_verilator",
            self.rtl_model_path / "testbench_verilator_dir" / "Vpicorv32_wrapper",
        ]

        for binary in possible_binaries:
            if binary.exists():
                self._sim_binary = binary.resolve()
                return self._sim_binary

        return None

    def _build_command(self, sim_binary: Path, hex_path: Path,
                       elf_path: Path, extra_args: Optional[List[str]] = None) -> List[str]:
        """Build simulation command."""
        cmd = [str(sim_binary)]

        # Common arguments for various testbenches
        cmd.extend(["+firmware=" + str(hex_path)])
        cmd.extend(["+max_cycles=" + str(self.config.rtl_timeout)])
        if extra_args:
            cmd.extend(extra_args)

        return cmd

    def capture_trace(self, program: UltimateProgram, output_dir: Path) -> RTLResult:
        """Run RTL simulation with trace/vcd enabled and store artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)
        hex_path = output_dir / "program.hex"
        elf_path = output_dir / "program.elf"
        write_hex(program, hex_path, base_address=0)
        self.elf_writer.write(program, elf_path)
        return self._run_simulation(hex_path, elf_path, output_dir, extra_args=["+trace", "+vcd"])

    def _parse_output(self, result: RTLResult) -> None:
        """Parse simulation output for cycle count and status."""
        output = result.raw_output

        # Look for cycle count
        cycle_match = re.search(r'(\d+)\s*cycles', output, re.IGNORECASE)
        if cycle_match:
            result.cycle_count = int(cycle_match.group(1))

        # Look for completion indicator
        if 'PASS' in output or 'SUCCESS' in output or 'completed' in output.lower():
            result.success = True
        elif 'TRAP' in output:
            # PicoRV32 testbench uses TRAP on program completion.
            result.success = True
            result.bug_detected = False
        elif 'FAIL' in output or 'ERROR' in output:
            result.success = False
            result.bug_detected = True

        # Look for timeout indicator
        if 'timeout' in output.lower() or 'hung' in output.lower():
            result.timeout = True
            result.bug_detected = True

    def build_simulation(self) -> Tuple[bool, str]:
        """
        Build the RTL simulation binary.

        Returns:
            (success, message)
        """
        if self.rtl_model_path is None:
            return False, "RTL model path not configured"

        # Check for Makefile
        makefile = self.rtl_model_path / "Makefile"
        if not makefile.exists():
            return False, f"No Makefile found at {self.rtl_model_path}"

        try:
            result = subprocess.run(
                ["make", "sim"],
                cwd=str(self.rtl_model_path),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for build
            )

            if result.returncode == 0:
                return True, "Build successful"
            else:
                return False, f"Build failed: {result.stderr}"

        except Exception as e:
            return False, str(e)

    def check_verilator_available(self) -> Tuple[bool, str]:
        """
        Check if Verilator is available.

        Returns:
            (available, message)
        """
        try:
            result = subprocess.run(
                [str(self.verilator_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                version = result.stdout.strip()
                # Check version >= 5.0
                match = re.search(r'(\d+)\.(\d+)', version)
                if match:
                    major = int(match.group(1))
                    if major >= 5:
                        return True, f"Verilator {version}"
                    else:
                        return False, f"Verilator {version} (need >= 5.0)"
                return True, version
            else:
                return False, f"Verilator error: {result.stderr}"

        except FileNotFoundError:
            return False, f"Verilator not found at {self.verilator_path}"
        except Exception as e:
            return False, str(e)


class MockRTLRunner:
    """
    Mock RTL runner for testing without actual RTL simulation.
    """

    def __init__(self, config: FuzzerConfig):
        """Initialize mock RTL runner."""
        self.config = config
        self._bug_probability = 0.01  # 1% chance of "detecting" a bug

    def run(self, program: UltimateProgram) -> RTLResult:
        """Run mock simulation."""
        import random

        result = RTLResult()
        result.success = True
        result.cycle_count = random.randint(1000, 50000)

        # Occasionally simulate a bug
        if random.random() < self._bug_probability:
            result.timeout = True
            result.bug_detected = True
            result.error_message = "Mock bug detected"
            result.success = False

        return result
