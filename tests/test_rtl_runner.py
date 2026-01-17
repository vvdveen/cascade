"""
Tests for RTL runner behavior and hex generation.
"""

from dataclasses import dataclass

from cascade.config import FuzzerConfig, CPUConfig, Extension
from cascade.execution.rtl_runner import RTLRunner, RTLResult
from cascade.execution.elf_writer import write_hex


def test_rtl_runner_parses_trap_success():
    """TRAP output should be treated as successful completion."""
    config = FuzzerConfig(cpu=CPUConfig(name="test", xlen=32, extensions={Extension.I}))
    runner = RTLRunner(config)
    result = RTLResult(raw_output="TRAP after 123 clock cycles\n")
    runner._parse_output(result)

    assert result.success is True
    assert result.bug_detected is False


def test_rtl_runner_parses_error_as_bug():
    """ERROR output should be treated as a bug."""
    config = FuzzerConfig(cpu=CPUConfig(name="test", xlen=32, extensions={Extension.I}))
    runner = RTLRunner(config)
    result = RTLResult(raw_output="ERROR: simulation failed\n")
    runner._parse_output(result)

    assert result.success is False
    assert result.bug_detected is True


def test_write_hex_base_address_override(tmp_path):
    """write_hex should honor base_address for memory init."""
    @dataclass
    class DummyProgram:
        code_start: int = 0x80000000

        def to_bytes(self) -> bytes:
            return b"\x13\x00\x00\x00"

    output = tmp_path / "program.hex"
    write_hex(DummyProgram(), output, base_address=0)

    first_line = output.read_text().splitlines()[0]
    assert first_line == "@00000000"
