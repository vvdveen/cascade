"""
Integration tests for RTL simulation.
"""

from pathlib import Path
import os
import tempfile

import pytest

from cascade.config import FuzzerConfig, CPUConfig, Extension
from cascade.execution.rtl_runner import RTLRunner
from cascade.execution.elf_writer import write_hex
from cascade.generator.basic_block import BasicBlock
from cascade.generator.ultimate import UltimateProgram
from cascade.isa.encoding import EncodedInstruction
from cascade.isa.instructions import ADDI, EBREAK, JAL, LW


def _get_rtl_runner() -> RTLRunner:
    rtl_path = Path(os.environ.get("CASCADE_RTL_PATH", "deps/picorv32"))
    config = FuzzerConfig(
        cpu=CPUConfig(name="picorv32", xlen=32, extensions={Extension.I, Extension.M}),
        rtl_model_path=rtl_path,
        rtl_timeout=2000,
    )
    runner = RTLRunner(config)
    if runner._get_sim_binary() is None:
        ok, msg = runner.build_simulation()
        if not ok:
            pytest.fail(f"RTL simulation binary not available: {msg}")
    return runner


def _make_program(runner: RTLRunner, instructions: list[EncodedInstruction]) -> UltimateProgram:
    code_start = runner.config.memory.code_start
    block = BasicBlock(start_addr=code_start, block_id=0)
    block.instructions = instructions[:-1]
    block.terminator = instructions[-1]
    return UltimateProgram(
        blocks=[block],
        entry_addr=code_start,
        code_start=code_start,
        code_end=code_start + len(instructions) * 4,
        data_start=runner.config.memory.data_start,
        data_end=runner.config.memory.data_start + runner.config.memory.data_size,
    )


def _run_program(runner: RTLRunner, program: UltimateProgram, extra_args: list[str] | None = None):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        hex_path = tmpdir / "program.hex"
        elf_path = tmpdir / "program.elf"
        write_hex(program, hex_path, base_address=0)
        runner.elf_writer.write(program, elf_path)
        return runner._run_simulation(hex_path, elf_path, tmpdir, extra_args=extra_args)


def test_rtl_program_completes_without_timeout():
    runner = _get_rtl_runner()
    program = _make_program(
        runner,
        [
            EncodedInstruction(ADDI, rd=1, rs1=0, imm=0),
            EncodedInstruction(EBREAK),
        ],
    )

    result = _run_program(runner, program, extra_args=["+noerror"])

    assert result.timeout is False
    assert result.success is True
    assert result.bug_detected is False


def test_rtl_program_exception_traps():
    runner = _get_rtl_runner()
    program = _make_program(
        runner,
        [
            EncodedInstruction(LW, rd=1, rs1=0, imm=1),
            EncodedInstruction(EBREAK),
        ],
    )

    result = _run_program(runner, program)

    assert result.timeout is False
    assert result.success is True


def test_rtl_program_timeout_is_detected():
    runner = _get_rtl_runner()
    program = _make_program(
        runner,
        [
            EncodedInstruction(ADDI, rd=1, rs1=0, imm=0),
            EncodedInstruction(JAL, rd=0, imm=0),
        ],
    )

    result = _run_program(runner, program)

    assert result.timeout is True
    assert result.bug_detected is True
