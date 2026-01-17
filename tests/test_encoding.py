"""
Tests for instruction encoding/decoding.
"""

import pytest
from cascade.isa.instructions import (
    ADD, SUB, AND, OR, XOR, SLL, SRL, SRA,
    ADDI, ANDI, ORI, XORI, SLTI, SLTIU, SLLI, SRLI, SRAI,
    LUI, AUIPC,
    LW, LH, LB, LHU, LBU,
    SW, SH, SB,
    BEQ, BNE, BLT, BGE, BLTU, BGEU,
    JAL, JALR,
)
from cascade.isa.encoding import (
    EncodedInstruction, encode_instruction, decode_instruction,
    nop, lui, addi, add, xor, jal, jalr, beq, bne, lw, sw,
)


class TestRTypeEncoding:
    """Test R-type instruction encoding."""

    def test_add_encoding(self):
        """Test ADD instruction encoding."""
        enc = EncodedInstruction(ADD, rd=1, rs1=2, rs2=3)
        binary = enc.binary

        # ADD: opcode=0110011, funct3=000, funct7=0000000
        assert binary & 0x7F == 0b0110011  # opcode
        assert (binary >> 7) & 0x1F == 1   # rd
        assert (binary >> 12) & 0x7 == 0   # funct3
        assert (binary >> 15) & 0x1F == 2  # rs1
        assert (binary >> 20) & 0x1F == 3  # rs2
        assert (binary >> 25) & 0x7F == 0  # funct7

    def test_sub_encoding(self):
        """Test SUB instruction encoding."""
        enc = EncodedInstruction(SUB, rd=5, rs1=6, rs2=7)
        binary = enc.binary

        assert binary & 0x7F == 0b0110011     # opcode
        assert (binary >> 25) & 0x7F == 0b0100000  # funct7 for SUB

    def test_and_or_xor(self):
        """Test AND, OR, XOR encoding."""
        and_enc = EncodedInstruction(AND, rd=1, rs1=2, rs2=3)
        or_enc = EncodedInstruction(OR, rd=1, rs1=2, rs2=3)
        xor_enc = EncodedInstruction(XOR, rd=1, rs1=2, rs2=3)

        # Check funct3 values
        assert (and_enc.binary >> 12) & 0x7 == 0b111
        assert (or_enc.binary >> 12) & 0x7 == 0b110
        assert (xor_enc.binary >> 12) & 0x7 == 0b100


class TestITypeEncoding:
    """Test I-type instruction encoding."""

    def test_addi_encoding(self):
        """Test ADDI instruction encoding."""
        enc = EncodedInstruction(ADDI, rd=1, rs1=2, imm=100)
        binary = enc.binary

        assert binary & 0x7F == 0b0010011  # opcode
        assert (binary >> 7) & 0x1F == 1   # rd
        assert (binary >> 12) & 0x7 == 0   # funct3
        assert (binary >> 15) & 0x1F == 2  # rs1
        assert (binary >> 20) & 0xFFF == 100  # imm

    def test_addi_negative_imm(self):
        """Test ADDI with negative immediate."""
        enc = EncodedInstruction(ADDI, rd=1, rs1=0, imm=-1)
        binary = enc.binary

        # -1 as 12-bit signed = 0xFFF
        assert (binary >> 20) & 0xFFF == 0xFFF

    def test_load_encoding(self):
        """Test load instruction encoding."""
        enc = EncodedInstruction(LW, rd=1, rs1=2, imm=8)
        binary = enc.binary

        assert binary & 0x7F == 0b0000011  # load opcode
        assert (binary >> 12) & 0x7 == 0b010  # funct3 for LW


class TestSTypeEncoding:
    """Test S-type instruction encoding."""

    def test_sw_encoding(self):
        """Test SW instruction encoding."""
        enc = EncodedInstruction(SW, rs1=2, rs2=1, imm=8)
        binary = enc.binary

        assert binary & 0x7F == 0b0100011  # store opcode
        assert (binary >> 12) & 0x7 == 0b010  # funct3 for SW

        # Extract immediate
        imm_4_0 = (binary >> 7) & 0x1F
        imm_11_5 = (binary >> 25) & 0x7F
        imm = (imm_11_5 << 5) | imm_4_0
        assert imm == 8


class TestBTypeEncoding:
    """Test B-type instruction encoding."""

    def test_beq_encoding(self):
        """Test BEQ instruction encoding."""
        enc = EncodedInstruction(BEQ, rs1=1, rs2=2, imm=8)
        binary = enc.binary

        assert binary & 0x7F == 0b1100011  # branch opcode
        assert (binary >> 12) & 0x7 == 0b000  # funct3 for BEQ

    def test_branch_offset(self):
        """Test branch offset encoding."""
        # Branch to +16
        enc = EncodedInstruction(BNE, rs1=1, rs2=2, imm=16)
        binary = enc.binary

        # Decode the offset
        imm_11 = (binary >> 7) & 0x1
        imm_4_1 = (binary >> 8) & 0xF
        imm_10_5 = (binary >> 25) & 0x3F
        imm_12 = (binary >> 31) & 0x1

        offset = (imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1)
        assert offset == 16


class TestUTypeEncoding:
    """Test U-type instruction encoding."""

    def test_lui_encoding(self):
        """Test LUI instruction encoding."""
        enc = EncodedInstruction(LUI, rd=1, imm=0x12345000)
        binary = enc.binary

        assert binary & 0x7F == 0b0110111  # LUI opcode
        assert (binary >> 7) & 0x1F == 1   # rd
        assert binary & 0xFFFFF000 == 0x12345000


class TestJTypeEncoding:
    """Test J-type instruction encoding."""

    def test_jal_encoding(self):
        """Test JAL instruction encoding."""
        enc = EncodedInstruction(JAL, rd=1, imm=100)
        binary = enc.binary

        assert binary & 0x7F == 0b1101111  # JAL opcode
        assert (binary >> 7) & 0x1F == 1   # rd


class TestConvenienceFunctions:
    """Test convenience encoding functions."""

    def test_nop(self):
        """Test NOP generation."""
        enc = nop()
        assert enc.instruction.name == 'addi'
        assert enc.rd == 0
        assert enc.rs1 == 0
        assert enc.imm == 0

    def test_lui(self):
        """Test LUI generation."""
        enc = lui(1, 0x12345000)
        assert enc.instruction.name == 'lui'
        assert enc.rd == 1
        assert enc.imm == 0x12345000

    def test_addi(self):
        """Test ADDI generation."""
        enc = addi(1, 2, 100)
        assert enc.instruction.name == 'addi'
        assert enc.rd == 1
        assert enc.rs1 == 2
        assert enc.imm == 100

    def test_jal(self):
        """Test JAL generation."""
        enc = jal(1, 100)
        assert enc.instruction.name == 'jal'
        assert enc.rd == 1
        assert enc.imm == 100


class TestAssemblyOutput:
    """Test assembly string generation."""

    def test_r_type_asm(self):
        """Test R-type assembly output."""
        enc = EncodedInstruction(ADD, rd=1, rs1=2, rs2=3)
        assert enc.to_asm() == "add x1, x2, x3"

    def test_i_type_asm(self):
        """Test I-type assembly output."""
        enc = EncodedInstruction(ADDI, rd=1, rs1=2, imm=100)
        assert enc.to_asm() == "addi x1, x2, 100"

    def test_load_asm(self):
        """Test load assembly output."""
        enc = EncodedInstruction(LW, rd=1, rs1=2, imm=8)
        assert enc.to_asm() == "lw x1, 8(x2)"

    def test_store_asm(self):
        """Test store assembly output."""
        enc = EncodedInstruction(SW, rs1=2, rs2=1, imm=8)
        assert enc.to_asm() == "sw x1, 8(x2)"

    def test_branch_asm(self):
        """Test branch assembly output."""
        enc = EncodedInstruction(BEQ, rs1=1, rs2=2, imm=16)
        assert enc.to_asm() == "beq x1, x2, 16"

    def test_jal_asm(self):
        """Test JAL assembly output."""
        enc = EncodedInstruction(JAL, rd=1, imm=100)
        assert enc.to_asm() == "jal x1, 100"
