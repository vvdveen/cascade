"""
Tests for ISS/RTL trace outputs in bug and good run directories.
"""

from pathlib import Path

from cascade.config import FuzzerConfig, CPUConfig, Extension
from cascade.execution.iss_runner import ISSResult
from cascade.execution.rtl_runner import RTLResult
from cascade.generator.basic_block import BasicBlock
from cascade.generator.intermediate import IntermediateProgram, ProgramDescriptor
from cascade.generator.ultimate import UltimateProgram, ISSFeedback
from cascade.isa.encoding import nop
from cascade.fuzzer import Fuzzer
from cascade.reduction.reducer import ReductionResult


class FakeISSRunner:
    """Fake ISS runner that returns scripted results and traces."""

    def __init__(self, results, trace_pcs):
        self._results = list(results)
        self._trace_pcs = list(trace_pcs)
        self.isa_string = "rv32im"

    def run(self, _program, collect_feedback=True):
        if not self._results:
            raise AssertionError("ISS runner called more times than expected")
        return self._results.pop(0)

    def run_trace(self, _program):
        return True, list(self._trace_pcs), "trace"


class FakeRTLRunner:
    """Fake RTL runner that writes a minimal VCD with instr_addr."""

    def __init__(self, result, vcd_pcs):
        self._result = result
        self._vcd_pcs = list(vcd_pcs)
        self.config = type("Config", (), {"rtl_timeout": 0})()

    def run(self, _program):
        return self._result

    def capture_trace(self, _program, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        vcd_path = output_dir / "testbench.vcd"
        _write_vcd_with_instr_addr(vcd_path, self._vcd_pcs)
        return self._result

    def _get_sim_binary(self):
        return None


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


class FakeReducer:
    """Fake reducer to avoid heavy reduction work in tests."""

    def reduce(self, program, feedback=None):
        return ReductionResult(original_size=1, reduced_size=0, reduced_program=None)


def _write_vcd_with_instr_addr(path: Path, pcs):
    header = [
        "$version test $end",
        "$timescale 1ns $end",
        "$scope module testbench $end",
        "$var wire 32 H instr_addr [31:0] $end",
        "$upscope $end",
        "$enddefinitions $end",
    ]
    lines = header[:]
    time = 0
    for pc in pcs:
        lines.append(f"#{time}")
        lines.append(f"b{pc:032b} H")
        time += 1
    path.write_text("\n".join(lines) + "\n")


def _make_intermediate_program(seed=123) -> IntermediateProgram:
    block = BasicBlock(start_addr=0x80000000, block_id=0)
    block.instructions = [nop()]
    block.terminator = nop()
    return IntermediateProgram(
        blocks=[block],
        entry_addr=0x80000000,
        code_start=0x80000000,
        code_end=block.end_addr,
        data_start=0x80010000,
        data_end=0x80010000,
        descriptor=ProgramDescriptor(seed=seed, num_blocks=1),
    )


def _make_ultimate_program() -> UltimateProgram:
    block = BasicBlock(start_addr=0x80000000, block_id=0)
    block.instructions = [nop()]
    block.terminator = nop()
    return UltimateProgram(
        blocks=[block],
        entry_addr=0x80000000,
        code_start=0x80000000,
        code_end=block.end_addr,
        data_start=0x80010000,
        data_end=0x80010000,
    )


def _assert_trace_files(run_dir: Path):
    assert (run_dir / "iss_trace_intermediate.txt").exists()
    assert (run_dir / "iss_trace_ultimate.txt").exists()
    assert (run_dir / "rtl_trace_pc.txt").exists()
    assert (run_dir / "pc_trace_compare.txt").exists()
    assert (run_dir / "ultimate.vcd").exists()


def test_trace_outputs_for_bug_run(tmp_path):
    config = FuzzerConfig(
        cpu=CPUConfig(name="test", xlen=32, extensions={Extension.I}),
        output_dir=tmp_path,
    )
    fuzzer = Fuzzer(config)

    intermediate = _make_intermediate_program(seed=1)
    ultimate = _make_ultimate_program()

    fuzzer.intermediate_gen = FakeIntermediateGen(intermediate)
    fuzzer.ultimate_gen = FakeUltimateGen(ultimate)
    fuzzer.reducer = FakeReducer()

    iss_ok = ISSResult(success=True, timeout=False, feedback=ISSFeedback())
    ultimate_ok = ISSResult(success=True, timeout=False)
    fuzzer.iss_runner = FakeISSRunner([iss_ok, ultimate_ok], [0x80000000, 0x80000004])

    rtl_result = RTLResult(success=False, timeout=True, bug_detected=True)
    fuzzer.rtl_runner = FakeRTLRunner(rtl_result, [0x80000000, 0x80000004, 0x80000008])

    fuzzer._fuzz_iteration(0)

    bug_dirs = list((tmp_path / "test" / "bugs").glob("bug_*"))
    assert len(bug_dirs) == 1
    _assert_trace_files(bug_dirs[0])


def test_trace_outputs_for_good_run(tmp_path):
    config = FuzzerConfig(
        cpu=CPUConfig(name="test", xlen=32, extensions={Extension.I}),
        output_dir=tmp_path,
    )
    fuzzer = Fuzzer(config)

    intermediate = _make_intermediate_program(seed=2)
    ultimate = _make_ultimate_program()

    fuzzer.intermediate_gen = FakeIntermediateGen(intermediate)
    fuzzer.ultimate_gen = FakeUltimateGen(ultimate)

    iss_ok = ISSResult(success=True, timeout=False, feedback=ISSFeedback())
    ultimate_ok = ISSResult(success=True, timeout=False)
    fuzzer.iss_runner = FakeISSRunner([iss_ok, ultimate_ok], [0x80000000, 0x80000004])

    rtl_result = RTLResult(success=True, timeout=False, bug_detected=False)
    fuzzer.rtl_runner = FakeRTLRunner(rtl_result, [0x80000000, 0x80000004, 0x80000008])

    fuzzer._fuzz_iteration(0)

    good_dirs = list((tmp_path / "test" / "good").glob("good_*"))
    assert len(good_dirs) == 1
    _assert_trace_files(good_dirs[0])


def test_trace_outputs_for_rtl_error(tmp_path):
    config = FuzzerConfig(
        cpu=CPUConfig(name="test", xlen=32, extensions={Extension.I}),
        output_dir=tmp_path,
    )
    fuzzer = Fuzzer(config)

    intermediate = _make_intermediate_program(seed=3)
    ultimate = _make_ultimate_program()

    fuzzer.intermediate_gen = FakeIntermediateGen(intermediate)
    fuzzer.ultimate_gen = FakeUltimateGen(ultimate)
    fuzzer.reducer = FakeReducer()

    iss_ok = ISSResult(success=True, timeout=False, feedback=ISSFeedback())
    ultimate_ok = ISSResult(success=True, timeout=False)
    fuzzer.iss_runner = FakeISSRunner([iss_ok, ultimate_ok], [0x80000000, 0x80000004])

    rtl_result = RTLResult(success=False, timeout=False, bug_detected=False)
    fuzzer.rtl_runner = FakeRTLRunner(rtl_result, [0x80000000, 0x80000004, 0x80000008])

    fuzzer._fuzz_iteration(0)

    error_dirs = list((tmp_path / "test" / "errors").glob("rtl_error_*"))
    assert len(error_dirs) == 1
    _assert_trace_files(error_dirs[0])
