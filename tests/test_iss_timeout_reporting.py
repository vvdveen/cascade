"""
Tests for ISS timeout reporting.
"""

from pathlib import Path

from cascade.config import FuzzerConfig, CPUConfig, Extension
from cascade.execution.iss_runner import ISSResult
from cascade.generator.intermediate import IntermediateProgram, ProgramDescriptor
from cascade.generator.basic_block import BasicBlock
from cascade.generator.ultimate import UltimateProgram, ISSFeedback
from cascade.isa.encoding import nop
from cascade.fuzzer import Fuzzer


class FakeISSRunner:
    """Fake ISS runner with scripted results."""

    def __init__(self, results):
        self._results = list(results)
        self.isa_string = "rv32im"

    def run(self, _program, collect_feedback=True):
        if not self._results:
            raise AssertionError("ISS runner called more times than expected")
        return self._results.pop(0)


class FakeRTLRunner:
    """Fake RTL runner to assert it is not invoked."""

    def __init__(self):
        self.called = False

    def run(self, _program):
        self.called = True
        raise AssertionError("RTL runner should not be called")


class FakeUltimateGen:
    """Fake ultimate generator."""

    def __init__(self, ultimate_program):
        self._ultimate = ultimate_program

    def generate(self, _intermediate, _feedback):
        return self._ultimate


class FakeIntermediateGen:
    """Fake intermediate generator."""

    def __init__(self, program):
        self._program = program

    def generate(self, seed=None):
        return self._program


def _make_intermediate_program(seed=123) -> IntermediateProgram:
    block = BasicBlock(start_addr=0x80000000, block_id=0)
    block.instructions = [nop()]
    block.terminator = nop()

    program = IntermediateProgram(
        blocks=[block],
        entry_addr=0x80000000,
        code_start=0x80000000,
        code_end=block.end_addr,
        data_start=0x80010000,
        data_end=0x80010000,
        descriptor=ProgramDescriptor(seed=seed, num_blocks=1),
    )
    return program


def _make_ultimate_program() -> UltimateProgram:
    block = BasicBlock(start_addr=0x80000000, block_id=0)
    block.instructions = [nop()]
    block.terminator = nop()

    program = UltimateProgram(
        blocks=[block],
        entry_addr=0x80000000,
        code_start=0x80000000,
        code_end=block.end_addr,
        data_start=0x80010000,
        data_end=0x80010000,
    )
    return program


def test_intermediate_iss_timeout_creates_error_report(tmp_path):
    config = FuzzerConfig(
        cpu=CPUConfig(name="test", xlen=32, extensions={Extension.I}),
        output_dir=tmp_path,
    )
    fuzzer = Fuzzer(config)

    program = _make_intermediate_program(seed=42)
    fuzzer.intermediate_gen = FakeIntermediateGen(program)
    fuzzer.ultimate_gen = FakeUltimateGen(_make_ultimate_program())
    fuzzer.rtl_runner = FakeRTLRunner()

    iss_result = ISSResult(success=False, timeout=True, error_message="timeout", raw_output="ISS timeout")
    fuzzer.iss_runner = FakeISSRunner([iss_result])

    fuzzer._fuzz_iteration(0)

    error_dirs = list((tmp_path / "test" / "errors").glob("iss_timeout_intermediate_*"))
    assert len(error_dirs) == 1
    error_dir = error_dirs[0]
    assert (error_dir / "metadata.txt").exists()
    assert (error_dir / "ultimate.elf").exists()
    assert (error_dir / "intermediate.elf").exists()
    assert (error_dir / "ultimate.S").exists()
    assert (error_dir / "rerun_iss.sh").exists()

    metadata = (error_dir / "metadata.txt").read_text()
    assert "ISS timeout: True" in metadata


def test_ultimate_iss_timeout_creates_error_report_and_skips_rtl(tmp_path):
    config = FuzzerConfig(
        cpu=CPUConfig(name="test", xlen=32, extensions={Extension.I}),
        output_dir=tmp_path,
    )
    fuzzer = Fuzzer(config)

    intermediate = _make_intermediate_program(seed=99)
    ultimate = _make_ultimate_program()
    fuzzer.intermediate_gen = FakeIntermediateGen(intermediate)
    fuzzer.ultimate_gen = FakeUltimateGen(ultimate)
    fuzzer.rtl_runner = FakeRTLRunner()

    iss_ok = ISSResult(success=True, timeout=False, feedback=ISSFeedback())
    iss_timeout = ISSResult(success=False, timeout=True, error_message="timeout", raw_output="ISS timeout")
    fuzzer.iss_runner = FakeISSRunner([iss_ok, iss_timeout])

    fuzzer._fuzz_iteration(0)

    error_dirs = list((tmp_path / "test" / "errors").glob("iss_timeout_ultimate_*"))
    assert len(error_dirs) == 1
    error_dir = error_dirs[0]
    assert (error_dir / "metadata.txt").exists()
    assert (error_dir / "ultimate.elf").exists()
    assert (error_dir / "intermediate.elf").exists()
    assert (error_dir / "ultimate.S").exists()
    assert (error_dir / "rerun_iss.sh").exists()

    metadata = (error_dir / "metadata.txt").read_text()
    assert "ISS timeout: True" in metadata
