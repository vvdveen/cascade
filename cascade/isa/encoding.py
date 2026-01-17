"""
RISC-V instruction encoding and decoding.
"""

from dataclasses import dataclass
from typing import Optional
from .instructions import Instruction, InstructionFormat


@dataclass
class EncodedInstruction:
    """An encoded RISC-V instruction with its operands."""
    instruction: Instruction
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0
    csr: int = 0  # For CSR instructions

    @property
    def binary(self) -> int:
        """Encode instruction to 32-bit binary."""
        return encode_instruction(self)

    def to_bytes(self) -> bytes:
        """Return instruction as little-endian bytes."""
        return self.binary.to_bytes(4, 'little')

    def to_asm(self) -> str:
        """Return assembly representation."""
        instr = self.instruction
        name = instr.name

        if instr.format == InstructionFormat.R:
            return f"{name} x{self.rd}, x{self.rs1}, x{self.rs2}"
        elif instr.format == InstructionFormat.I:
            if instr.is_load:
                return f"{name} x{self.rd}, {self.imm}(x{self.rs1})"
            elif instr.reads_csr:
                return f"{name} x{self.rd}, {self.csr}, x{self.rs1}"
            elif name == "jalr":
                return f"{name} x{self.rd}, x{self.rs1}, {self.imm}"
            else:
                return f"{name} x{self.rd}, x{self.rs1}, {self.imm}"
        elif instr.format == InstructionFormat.S:
            return f"{name} x{self.rs2}, {self.imm}(x{self.rs1})"
        elif instr.format == InstructionFormat.B:
            return f"{name} x{self.rs1}, x{self.rs2}, {self.imm}"
        elif instr.format == InstructionFormat.U:
            return f"{name} x{self.rd}, {self.imm >> 12}"
        elif instr.format == InstructionFormat.J:
            return f"{name} x{self.rd}, {self.imm}"
        else:
            return name


def sign_extend(value: int, bits: int) -> int:
    """Sign-extend a value to 32 bits."""
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def encode_r_type(opcode: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int) -> int:
    """Encode R-type instruction."""
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def encode_i_type(opcode: int, rd: int, funct3: int, rs1: int, imm: int) -> int:
    """Encode I-type instruction."""
    imm12 = imm & 0xFFF
    return (imm12 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def encode_s_type(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    """Encode S-type instruction."""
    imm12 = imm & 0xFFF
    imm_11_5 = (imm12 >> 5) & 0x7F
    imm_4_0 = imm12 & 0x1F
    return (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | opcode


def encode_b_type(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    """Encode B-type instruction (branch offset is in multiples of 2)."""
    imm13 = imm & 0x1FFE  # Bit 0 is always 0
    imm_12 = (imm13 >> 12) & 0x1
    imm_11 = (imm13 >> 11) & 0x1
    imm_10_5 = (imm13 >> 5) & 0x3F
    imm_4_1 = (imm13 >> 1) & 0xF
    return ((imm_12 << 31) | (imm_10_5 << 25) | (rs2 << 20) | (rs1 << 15) |
            (funct3 << 12) | (imm_4_1 << 8) | (imm_11 << 7) | opcode)


def encode_u_type(opcode: int, rd: int, imm: int) -> int:
    """Encode U-type instruction (upper 20 bits)."""
    imm20 = imm & 0xFFFFF000
    return imm20 | (rd << 7) | opcode


def encode_j_type(opcode: int, rd: int, imm: int) -> int:
    """Encode J-type instruction (jump offset is in multiples of 2)."""
    imm21 = imm & 0x1FFFFE  # Bit 0 is always 0
    imm_20 = (imm21 >> 20) & 0x1
    imm_19_12 = (imm21 >> 12) & 0xFF
    imm_11 = (imm21 >> 11) & 0x1
    imm_10_1 = (imm21 >> 1) & 0x3FF
    return ((imm_20 << 31) | (imm_10_1 << 21) | (imm_11 << 20) |
            (imm_19_12 << 12) | (rd << 7) | opcode)


def encode_instruction(enc: EncodedInstruction) -> int:
    """Encode an instruction to its 32-bit binary representation."""
    instr = enc.instruction
    fmt = instr.format
    opcode = instr.opcode
    funct3 = instr.funct3 or 0
    funct7 = instr.funct7 or 0

    if fmt == InstructionFormat.R:
        return encode_r_type(opcode, enc.rd, funct3, enc.rs1, enc.rs2, funct7)
    elif fmt == InstructionFormat.I:
        # Handle shift instructions with funct7
        if instr.name in ('slli', 'srli', 'srai'):
            imm = (funct7 << 5) | (enc.imm & 0x1F)
            return encode_i_type(opcode, enc.rd, funct3, enc.rs1, imm)
        # Handle CSR instructions
        elif instr.reads_csr or instr.writes_csr:
            return encode_i_type(opcode, enc.rd, funct3, enc.rs1, enc.csr)
        # Handle system instructions
        elif instr.name == 'ecall':
            return encode_i_type(opcode, 0, 0, 0, 0)
        elif instr.name == 'ebreak':
            return encode_i_type(opcode, 0, 0, 0, 1)
        elif instr.name == 'mret':
            return encode_i_type(opcode, 0, 0, 0, 0b001100000010)
        elif instr.name == 'sret':
            return encode_i_type(opcode, 0, 0, 0, 0b000100000010)
        elif instr.name == 'wfi':
            return encode_i_type(opcode, 0, 0, 0, 0b000100000101)
        else:
            return encode_i_type(opcode, enc.rd, funct3, enc.rs1, enc.imm)
    elif fmt == InstructionFormat.S:
        return encode_s_type(opcode, funct3, enc.rs1, enc.rs2, enc.imm)
    elif fmt == InstructionFormat.B:
        return encode_b_type(opcode, funct3, enc.rs1, enc.rs2, enc.imm)
    elif fmt == InstructionFormat.U:
        return encode_u_type(opcode, enc.rd, enc.imm)
    elif fmt == InstructionFormat.J:
        return encode_j_type(opcode, enc.rd, enc.imm)
    else:
        raise ValueError(f"Unknown instruction format: {fmt}")


def decode_r_type(binary: int) -> tuple:
    """Decode R-type instruction fields."""
    opcode = binary & 0x7F
    rd = (binary >> 7) & 0x1F
    funct3 = (binary >> 12) & 0x7
    rs1 = (binary >> 15) & 0x1F
    rs2 = (binary >> 20) & 0x1F
    funct7 = (binary >> 25) & 0x7F
    return opcode, rd, funct3, rs1, rs2, funct7


def decode_i_type(binary: int) -> tuple:
    """Decode I-type instruction fields."""
    opcode = binary & 0x7F
    rd = (binary >> 7) & 0x1F
    funct3 = (binary >> 12) & 0x7
    rs1 = (binary >> 15) & 0x1F
    imm = sign_extend((binary >> 20) & 0xFFF, 12)
    return opcode, rd, funct3, rs1, imm


def decode_s_type(binary: int) -> tuple:
    """Decode S-type instruction fields."""
    opcode = binary & 0x7F
    imm_4_0 = (binary >> 7) & 0x1F
    funct3 = (binary >> 12) & 0x7
    rs1 = (binary >> 15) & 0x1F
    rs2 = (binary >> 20) & 0x1F
    imm_11_5 = (binary >> 25) & 0x7F
    imm = sign_extend((imm_11_5 << 5) | imm_4_0, 12)
    return opcode, funct3, rs1, rs2, imm


def decode_b_type(binary: int) -> tuple:
    """Decode B-type instruction fields."""
    opcode = binary & 0x7F
    imm_11 = (binary >> 7) & 0x1
    imm_4_1 = (binary >> 8) & 0xF
    funct3 = (binary >> 12) & 0x7
    rs1 = (binary >> 15) & 0x1F
    rs2 = (binary >> 20) & 0x1F
    imm_10_5 = (binary >> 25) & 0x3F
    imm_12 = (binary >> 31) & 0x1
    imm = sign_extend((imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1), 13)
    return opcode, funct3, rs1, rs2, imm


def decode_u_type(binary: int) -> tuple:
    """Decode U-type instruction fields."""
    opcode = binary & 0x7F
    rd = (binary >> 7) & 0x1F
    imm = binary & 0xFFFFF000
    return opcode, rd, imm


def decode_j_type(binary: int) -> tuple:
    """Decode J-type instruction fields."""
    opcode = binary & 0x7F
    rd = (binary >> 7) & 0x1F
    imm_19_12 = (binary >> 12) & 0xFF
    imm_11 = (binary >> 20) & 0x1
    imm_10_1 = (binary >> 21) & 0x3FF
    imm_20 = (binary >> 31) & 0x1
    imm = sign_extend((imm_20 << 20) | (imm_19_12 << 12) | (imm_11 << 11) | (imm_10_1 << 1), 21)
    return opcode, rd, imm


def decode_instruction(binary: int, instruction: Instruction) -> EncodedInstruction:
    """Decode a 32-bit instruction given its Instruction type."""
    fmt = instruction.format

    if fmt == InstructionFormat.R:
        _, rd, _, rs1, rs2, _ = decode_r_type(binary)
        return EncodedInstruction(instruction, rd=rd, rs1=rs1, rs2=rs2)
    elif fmt == InstructionFormat.I:
        _, rd, _, rs1, imm = decode_i_type(binary)
        if instruction.reads_csr or instruction.writes_csr:
            return EncodedInstruction(instruction, rd=rd, rs1=rs1, csr=imm & 0xFFF)
        return EncodedInstruction(instruction, rd=rd, rs1=rs1, imm=imm)
    elif fmt == InstructionFormat.S:
        _, _, rs1, rs2, imm = decode_s_type(binary)
        return EncodedInstruction(instruction, rs1=rs1, rs2=rs2, imm=imm)
    elif fmt == InstructionFormat.B:
        _, _, rs1, rs2, imm = decode_b_type(binary)
        return EncodedInstruction(instruction, rs1=rs1, rs2=rs2, imm=imm)
    elif fmt == InstructionFormat.U:
        _, rd, imm = decode_u_type(binary)
        return EncodedInstruction(instruction, rd=rd, imm=imm)
    elif fmt == InstructionFormat.J:
        _, rd, imm = decode_j_type(binary)
        return EncodedInstruction(instruction, rd=rd, imm=imm)
    else:
        raise ValueError(f"Unknown instruction format: {fmt}")


# Convenience functions for creating common instructions
def nop() -> EncodedInstruction:
    """Create a NOP instruction (addi x0, x0, 0)."""
    from .instructions import ADDI
    return EncodedInstruction(ADDI, rd=0, rs1=0, imm=0)


def lui(rd: int, imm: int) -> EncodedInstruction:
    """Create LUI instruction."""
    from .instructions import LUI
    return EncodedInstruction(LUI, rd=rd, imm=imm & 0xFFFFF000)


def addi(rd: int, rs1: int, imm: int) -> EncodedInstruction:
    """Create ADDI instruction."""
    from .instructions import ADDI
    return EncodedInstruction(ADDI, rd=rd, rs1=rs1, imm=imm & 0xFFF)


def add(rd: int, rs1: int, rs2: int) -> EncodedInstruction:
    """Create ADD instruction."""
    from .instructions import ADD
    return EncodedInstruction(ADD, rd=rd, rs1=rs1, rs2=rs2)


def xor(rd: int, rs1: int, rs2: int) -> EncodedInstruction:
    """Create XOR instruction."""
    from .instructions import XOR
    return EncodedInstruction(XOR, rd=rd, rs1=rs1, rs2=rs2)


def jal(rd: int, imm: int) -> EncodedInstruction:
    """Create JAL instruction."""
    from .instructions import JAL
    return EncodedInstruction(JAL, rd=rd, imm=imm)


def jalr(rd: int, rs1: int, imm: int) -> EncodedInstruction:
    """Create JALR instruction."""
    from .instructions import JALR
    return EncodedInstruction(JALR, rd=rd, rs1=rs1, imm=imm)


def beq(rs1: int, rs2: int, imm: int) -> EncodedInstruction:
    """Create BEQ instruction."""
    from .instructions import BEQ
    return EncodedInstruction(BEQ, rs1=rs1, rs2=rs2, imm=imm)


def bne(rs1: int, rs2: int, imm: int) -> EncodedInstruction:
    """Create BNE instruction."""
    from .instructions import BNE
    return EncodedInstruction(BNE, rs1=rs1, rs2=rs2, imm=imm)


def lw(rd: int, rs1: int, imm: int) -> EncodedInstruction:
    """Create LW instruction."""
    from .instructions import LW
    return EncodedInstruction(LW, rd=rd, rs1=rs1, imm=imm)


def sw(rs1: int, rs2: int, imm: int) -> EncodedInstruction:
    """Create SW instruction."""
    from .instructions import SW
    return EncodedInstruction(SW, rs1=rs1, rs2=rs2, imm=imm)
