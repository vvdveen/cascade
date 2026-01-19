"""
Integration tests for RTL simulation.
"""

from pathlib import Path
import os
import tempfile

import pytest

from cascade.config import (
    FuzzerConfig, CPUConfig, Extension,
    PICORV32_CONFIG, KRONOS_CONFIG, KRONOS_MEMORY_LAYOUT,
)
from cascade.execution.rtl_runner import RTLRunner
from cascade.execution.elf_writer import write_hex
from cascade.generator.basic_block import BasicBlock
from cascade.generator.ultimate import UltimateProgram
from cascade.isa.encoding import EncodedInstruction
from cascade.isa.instructions import ADDI, EBREAK, JAL, LW, LUI, SW


def _get_rtl_runner() -> RTLRunner:
    rtl_path = Path(os.environ.get("CASCADE_RTL_PATH", "deps/picorv32"))
    cpu_name = os.environ.get("CASCADE_RTL_CPU", "picorv32")
    if cpu_name == "kronos":
        cpu = KRONOS_CONFIG
        memory = KRONOS_MEMORY_LAYOUT
    else:
        cpu = PICORV32_CONFIG
        memory = FuzzerConfig().memory
    config = FuzzerConfig(
        cpu=cpu,
        memory=memory,
        rtl_model_path=rtl_path,
        rtl_timeout=2000,
    )
    runner = RTLRunner(config)
    if runner._get_sim_binary() is None:
        ok, msg = runner.build_simulation()
        if not ok:
            pytest.fail(f"RTL simulation binary not available: {msg}")
    return runner


def _get_rtl_runner_for(cpu_name: str, rtl_path: Path) -> RTLRunner:
    if cpu_name == "kronos":
        cpu = KRONOS_CONFIG
        memory = KRONOS_MEMORY_LAYOUT
    else:
        cpu = PICORV32_CONFIG
        memory = FuzzerConfig().memory
    config = FuzzerConfig(
        cpu=cpu,
        memory=memory,
        rtl_model_path=rtl_path,
        rtl_timeout=2000,
    )
    runner = RTLRunner(config)
    if cpu_name == "kronos":
        ok, msg = runner.build_simulation()
        if not ok:
            pytest.skip(f"RTL simulation binary not available: {msg}")
    elif runner._get_sim_binary() is None:
        ok, msg = runner.build_simulation()
        if not ok:
            pytest.skip(f"RTL simulation binary not available: {msg}")
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


def _load_addr(reg: int, addr: int) -> list[EncodedInstruction]:
    upper = (addr + 0x800) >> 12
    lower = addr - (upper << 12)
    return [
        EncodedInstruction(LUI, rd=reg, imm=(upper & 0xFFFFF) << 12),
        EncodedInstruction(ADDI, rd=reg, rs1=reg, imm=lower & 0xFFF),
    ]


def _run_program(runner: RTLRunner, program: UltimateProgram, extra_args: list[str] | None = None):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        if runner.config.cpu.name == "kronos":
            return runner.run(program)
        hex_path = tmpdir / "program.hex"
        elf_path = tmpdir / "program.elf"
        write_hex(program, hex_path, base_address=0)
        runner.elf_writer.write(program, elf_path)
        return runner._run_simulation(hex_path, elf_path, tmpdir, extra_args=extra_args)


def test_rtl_program_completes_without_timeout():
    runner = _get_rtl_runner()
    if runner.config.cpu.name == "kronos":
        data_start = runner.config.memory.data_start
        program = _make_program(
            runner,
            [
                *_load_addr(2, data_start),
                EncodedInstruction(ADDI, rd=1, rs1=0, imm=1),
                EncodedInstruction(SW, rs1=2, rs2=1, imm=0),
                EncodedInstruction(EBREAK),
            ],
        )
    else:
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
    if runner.config.cpu.name == "kronos":
        data_start = runner.config.memory.data_start
        program = _make_program(
            runner,
            [
                EncodedInstruction(LW, rd=1, rs1=0, imm=1),
                *_load_addr(2, data_start),
                EncodedInstruction(ADDI, rd=1, rs1=0, imm=1),
                EncodedInstruction(SW, rs1=2, rs2=1, imm=0),
                EncodedInstruction(EBREAK),
            ],
        )
    else:
        program = _make_program(
            runner,
            [
                EncodedInstruction(LW, rd=1, rs1=0, imm=1),
                EncodedInstruction(EBREAK),
            ],
        )

    result = _run_program(runner, program)

    if runner.config.cpu.name == "kronos":
        assert result.timeout is True
        assert result.bug_detected is True
    else:
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


def test_rtl_trace_writes_vcd():
    runner = _get_rtl_runner()
    program = _make_program(
        runner,
        [
            EncodedInstruction(ADDI, rd=1, rs1=0, imm=0),
            EncodedInstruction(EBREAK),
        ],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        trace_dir = tmpdir / "rtl_trace"
        runner.capture_trace(program, trace_dir)
        assert any(trace_dir.glob("*.vcd"))


def test_kronos_rtl_program_completes_without_timeout():
    rtl_path = Path("deps/kronos")
    if not rtl_path.exists():
        pytest.skip("Kronos RTL path not available")
    runner = _get_rtl_runner_for("kronos", rtl_path)
    data_start = runner.config.memory.data_start
    program = _make_program(
        runner,
        [
            *_load_addr(2, data_start),
            EncodedInstruction(ADDI, rd=1, rs1=0, imm=1),
            EncodedInstruction(SW, rs1=2, rs2=1, imm=0),
            EncodedInstruction(EBREAK),
        ],
    )

    result = _run_program(runner, program)

    assert result.timeout is False
    assert result.success is True
    assert result.bug_detected is False
