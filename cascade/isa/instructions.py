"""
RISC-V instruction definitions.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Tuple


class InstructionFormat(Enum):
    """RISC-V instruction formats."""
    R = auto()   # Register-register
    I = auto()   # Immediate
    S = auto()   # Store
    B = auto()   # Branch
    U = auto()   # Upper immediate
    J = auto()   # Jump


class InstructionCategory(Enum):
    """Instruction categories for generation."""
    # Phase 1: rv32i base
    REGFSM = auto()    # Register lifecycle (lui, addi for setup)
    ALU = auto()       # Arithmetic/logic operations
    JAL = auto()       # Direct jumps
    JALR = auto()      # Indirect jumps
    BRANCH = auto()    # Conditional branches
    MEM = auto()       # Memory operations
    RDWRCSR = auto()   # CSR read/write
    FENCES = auto()    # Fences

    # Phase 2: M extension
    MULDIV = auto()    # Multiplication/division

    # Phase 3: Privileged
    TVECFSM = auto()   # Trap vector update
    PPFSM = auto()     # Previous privilege update
    EPCFSM = auto()    # Exception PC update
    MEDELEG = auto()   # Exception delegation
    EXCEPTION = auto() # Trigger exception
    DWNPRV = auto()    # Downward privilege

    # Phase 4: F/D extensions (future)
    FPUFSM = auto()    # FPU state
    MEMFPU = auto()    # Float memory
    FPU = auto()       # Float operations
    MEMFPUD = auto()   # Double memory
    FPUD = auto()      # Double operations

    # Phase 5: 64-bit (future)
    ALU64 = auto()
    MULDIV64 = auto()
    MEM64 = auto()
    AMO64 = auto()
    FPU64 = auto()
    FPUD64 = auto()


class ControlFlowType(Enum):
    """Control flow behavior of instruction."""
    STILL = auto()       # Does not change control flow
    HOPPING = auto()     # Changes control flow
    CF_AMBIGUOUS = auto() # Depends on register values


@dataclass
class Instruction:
    """RISC-V instruction definition."""
    name: str
    format: InstructionFormat
    category: InstructionCategory
    opcode: int
    funct3: Optional[int] = None
    funct7: Optional[int] = None

    # Control flow properties
    cf_type: ControlFlowType = ControlFlowType.STILL

    # Instruction properties
    reads_rs1: bool = False
    reads_rs2: bool = False
    writes_rd: bool = False

    # For memory instructions
    is_load: bool = False
    is_store: bool = False
    mem_size: int = 0  # 1, 2, 4, 8 bytes

    # For CSR instructions
    reads_csr: bool = False
    writes_csr: bool = False

    def is_cf_ambiguous(self) -> bool:
        """Check if control flow depends on register values."""
        return self.cf_type == ControlFlowType.CF_AMBIGUOUS

    def is_hopping(self) -> bool:
        """Check if instruction changes control flow."""
        return self.cf_type in (ControlFlowType.HOPPING, ControlFlowType.CF_AMBIGUOUS)


# RV32I Base Integer Instructions

# R-type ALU instructions
ADD = Instruction("add", InstructionFormat.R, InstructionCategory.ALU,
                  opcode=0b0110011, funct3=0b000, funct7=0b0000000,
                  reads_rs1=True, reads_rs2=True, writes_rd=True)
SUB = Instruction("sub", InstructionFormat.R, InstructionCategory.ALU,
                  opcode=0b0110011, funct3=0b000, funct7=0b0100000,
                  reads_rs1=True, reads_rs2=True, writes_rd=True)
SLL = Instruction("sll", InstructionFormat.R, InstructionCategory.ALU,
                  opcode=0b0110011, funct3=0b001, funct7=0b0000000,
                  reads_rs1=True, reads_rs2=True, writes_rd=True)
SLT = Instruction("slt", InstructionFormat.R, InstructionCategory.ALU,
                  opcode=0b0110011, funct3=0b010, funct7=0b0000000,
                  reads_rs1=True, reads_rs2=True, writes_rd=True)
SLTU = Instruction("sltu", InstructionFormat.R, InstructionCategory.ALU,
                   opcode=0b0110011, funct3=0b011, funct7=0b0000000,
                   reads_rs1=True, reads_rs2=True, writes_rd=True)
XOR = Instruction("xor", InstructionFormat.R, InstructionCategory.ALU,
                  opcode=0b0110011, funct3=0b100, funct7=0b0000000,
                  reads_rs1=True, reads_rs2=True, writes_rd=True)
SRL = Instruction("srl", InstructionFormat.R, InstructionCategory.ALU,
                  opcode=0b0110011, funct3=0b101, funct7=0b0000000,
                  reads_rs1=True, reads_rs2=True, writes_rd=True)
SRA = Instruction("sra", InstructionFormat.R, InstructionCategory.ALU,
                  opcode=0b0110011, funct3=0b101, funct7=0b0100000,
                  reads_rs1=True, reads_rs2=True, writes_rd=True)
OR = Instruction("or", InstructionFormat.R, InstructionCategory.ALU,
                 opcode=0b0110011, funct3=0b110, funct7=0b0000000,
                 reads_rs1=True, reads_rs2=True, writes_rd=True)
AND = Instruction("and", InstructionFormat.R, InstructionCategory.ALU,
                  opcode=0b0110011, funct3=0b111, funct7=0b0000000,
                  reads_rs1=True, reads_rs2=True, writes_rd=True)

# I-type ALU instructions
ADDI = Instruction("addi", InstructionFormat.I, InstructionCategory.ALU,
                   opcode=0b0010011, funct3=0b000,
                   reads_rs1=True, writes_rd=True)
SLTI = Instruction("slti", InstructionFormat.I, InstructionCategory.ALU,
                   opcode=0b0010011, funct3=0b010,
                   reads_rs1=True, writes_rd=True)
SLTIU = Instruction("sltiu", InstructionFormat.I, InstructionCategory.ALU,
                    opcode=0b0010011, funct3=0b011,
                    reads_rs1=True, writes_rd=True)
XORI = Instruction("xori", InstructionFormat.I, InstructionCategory.ALU,
                   opcode=0b0010011, funct3=0b100,
                   reads_rs1=True, writes_rd=True)
ORI = Instruction("ori", InstructionFormat.I, InstructionCategory.ALU,
                  opcode=0b0010011, funct3=0b110,
                  reads_rs1=True, writes_rd=True)
ANDI = Instruction("andi", InstructionFormat.I, InstructionCategory.ALU,
                   opcode=0b0010011, funct3=0b111,
                   reads_rs1=True, writes_rd=True)
SLLI = Instruction("slli", InstructionFormat.I, InstructionCategory.ALU,
                   opcode=0b0010011, funct3=0b001, funct7=0b0000000,
                   reads_rs1=True, writes_rd=True)
SRLI = Instruction("srli", InstructionFormat.I, InstructionCategory.ALU,
                   opcode=0b0010011, funct3=0b101, funct7=0b0000000,
                   reads_rs1=True, writes_rd=True)
SRAI = Instruction("srai", InstructionFormat.I, InstructionCategory.ALU,
                   opcode=0b0010011, funct3=0b101, funct7=0b0100000,
                   reads_rs1=True, writes_rd=True)

# U-type instructions (for register setup)
LUI = Instruction("lui", InstructionFormat.U, InstructionCategory.REGFSM,
                  opcode=0b0110111, writes_rd=True)
AUIPC = Instruction("auipc", InstructionFormat.U, InstructionCategory.REGFSM,
                    opcode=0b0010111, writes_rd=True)

# Load instructions
LB = Instruction("lb", InstructionFormat.I, InstructionCategory.MEM,
                 opcode=0b0000011, funct3=0b000,
                 reads_rs1=True, writes_rd=True, is_load=True, mem_size=1,
                 cf_type=ControlFlowType.CF_AMBIGUOUS)
LH = Instruction("lh", InstructionFormat.I, InstructionCategory.MEM,
                 opcode=0b0000011, funct3=0b001,
                 reads_rs1=True, writes_rd=True, is_load=True, mem_size=2,
                 cf_type=ControlFlowType.CF_AMBIGUOUS)
LW = Instruction("lw", InstructionFormat.I, InstructionCategory.MEM,
                 opcode=0b0000011, funct3=0b010,
                 reads_rs1=True, writes_rd=True, is_load=True, mem_size=4,
                 cf_type=ControlFlowType.CF_AMBIGUOUS)
LBU = Instruction("lbu", InstructionFormat.I, InstructionCategory.MEM,
                  opcode=0b0000011, funct3=0b100,
                  reads_rs1=True, writes_rd=True, is_load=True, mem_size=1,
                  cf_type=ControlFlowType.CF_AMBIGUOUS)
LHU = Instruction("lhu", InstructionFormat.I, InstructionCategory.MEM,
                  opcode=0b0000011, funct3=0b101,
                  reads_rs1=True, writes_rd=True, is_load=True, mem_size=2,
                  cf_type=ControlFlowType.CF_AMBIGUOUS)

# Store instructions
SB = Instruction("sb", InstructionFormat.S, InstructionCategory.MEM,
                 opcode=0b0100011, funct3=0b000,
                 reads_rs1=True, reads_rs2=True, is_store=True, mem_size=1)
SH = Instruction("sh", InstructionFormat.S, InstructionCategory.MEM,
                 opcode=0b0100011, funct3=0b001,
                 reads_rs1=True, reads_rs2=True, is_store=True, mem_size=2)
SW = Instruction("sw", InstructionFormat.S, InstructionCategory.MEM,
                 opcode=0b0100011, funct3=0b010,
                 reads_rs1=True, reads_rs2=True, is_store=True, mem_size=4)

# Branch instructions
BEQ = Instruction("beq", InstructionFormat.B, InstructionCategory.BRANCH,
                  opcode=0b1100011, funct3=0b000,
                  reads_rs1=True, reads_rs2=True,
                  cf_type=ControlFlowType.CF_AMBIGUOUS)
BNE = Instruction("bne", InstructionFormat.B, InstructionCategory.BRANCH,
                  opcode=0b1100011, funct3=0b001,
                  reads_rs1=True, reads_rs2=True,
                  cf_type=ControlFlowType.CF_AMBIGUOUS)
BLT = Instruction("blt", InstructionFormat.B, InstructionCategory.BRANCH,
                  opcode=0b1100011, funct3=0b100,
                  reads_rs1=True, reads_rs2=True,
                  cf_type=ControlFlowType.CF_AMBIGUOUS)
BGE = Instruction("bge", InstructionFormat.B, InstructionCategory.BRANCH,
                  opcode=0b1100011, funct3=0b101,
                  reads_rs1=True, reads_rs2=True,
                  cf_type=ControlFlowType.CF_AMBIGUOUS)
BLTU = Instruction("bltu", InstructionFormat.B, InstructionCategory.BRANCH,
                   opcode=0b1100011, funct3=0b110,
                   reads_rs1=True, reads_rs2=True,
                   cf_type=ControlFlowType.CF_AMBIGUOUS)
BGEU = Instruction("bgeu", InstructionFormat.B, InstructionCategory.BRANCH,
                   opcode=0b1100011, funct3=0b111,
                   reads_rs1=True, reads_rs2=True,
                   cf_type=ControlFlowType.CF_AMBIGUOUS)

# Jump instructions
JAL = Instruction("jal", InstructionFormat.J, InstructionCategory.JAL,
                  opcode=0b1101111, writes_rd=True,
                  cf_type=ControlFlowType.HOPPING)
JALR = Instruction("jalr", InstructionFormat.I, InstructionCategory.JALR,
                   opcode=0b1100111, funct3=0b000,
                   reads_rs1=True, writes_rd=True,
                   cf_type=ControlFlowType.CF_AMBIGUOUS)

# System instructions
ECALL = Instruction("ecall", InstructionFormat.I, InstructionCategory.EXCEPTION,
                    opcode=0b1110011, funct3=0b000,
                    cf_type=ControlFlowType.HOPPING)
EBREAK = Instruction("ebreak", InstructionFormat.I, InstructionCategory.EXCEPTION,
                     opcode=0b1110011, funct3=0b000,
                     cf_type=ControlFlowType.HOPPING)
MRET = Instruction("mret", InstructionFormat.I, InstructionCategory.DWNPRV,
                   opcode=0b1110011, funct3=0b000, funct7=0b0011000,
                   cf_type=ControlFlowType.HOPPING)
SRET = Instruction("sret", InstructionFormat.I, InstructionCategory.DWNPRV,
                   opcode=0b1110011, funct3=0b000, funct7=0b0001000,
                   cf_type=ControlFlowType.HOPPING)
WFI = Instruction("wfi", InstructionFormat.I, InstructionCategory.FENCES,
                  opcode=0b1110011, funct3=0b000, funct7=0b0001000)

# Fence instructions
FENCE = Instruction("fence", InstructionFormat.I, InstructionCategory.FENCES,
                    opcode=0b0001111, funct3=0b000)
FENCE_I = Instruction("fence.i", InstructionFormat.I, InstructionCategory.FENCES,
                      opcode=0b0001111, funct3=0b001)

# CSR instructions
CSRRW = Instruction("csrrw", InstructionFormat.I, InstructionCategory.RDWRCSR,
                    opcode=0b1110011, funct3=0b001,
                    reads_rs1=True, writes_rd=True, reads_csr=True, writes_csr=True)
CSRRS = Instruction("csrrs", InstructionFormat.I, InstructionCategory.RDWRCSR,
                    opcode=0b1110011, funct3=0b010,
                    reads_rs1=True, writes_rd=True, reads_csr=True, writes_csr=True)
CSRRC = Instruction("csrrc", InstructionFormat.I, InstructionCategory.RDWRCSR,
                    opcode=0b1110011, funct3=0b011,
                    reads_rs1=True, writes_rd=True, reads_csr=True, writes_csr=True)
CSRRWI = Instruction("csrrwi", InstructionFormat.I, InstructionCategory.RDWRCSR,
                     opcode=0b1110011, funct3=0b101,
                     writes_rd=True, reads_csr=True, writes_csr=True)
CSRRSI = Instruction("csrrsi", InstructionFormat.I, InstructionCategory.RDWRCSR,
                     opcode=0b1110011, funct3=0b110,
                     writes_rd=True, reads_csr=True, writes_csr=True)
CSRRCI = Instruction("csrrci", InstructionFormat.I, InstructionCategory.RDWRCSR,
                     opcode=0b1110011, funct3=0b111,
                     writes_rd=True, reads_csr=True, writes_csr=True)


# RV32I instruction list
RV32I_INSTRUCTIONS: List[Instruction] = [
    # ALU R-type
    ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND,
    # ALU I-type
    ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI,
    # Upper immediate
    LUI, AUIPC,
    # Loads
    LB, LH, LW, LBU, LHU,
    # Stores
    SB, SH, SW,
    # Branches
    BEQ, BNE, BLT, BGE, BLTU, BGEU,
    # Jumps
    JAL, JALR,
    # System
    ECALL, EBREAK, MRET, SRET, WFI,
    # Fences
    FENCE, FENCE_I,
    # CSR
    CSRRW, CSRRS, CSRRC, CSRRWI, CSRRSI, CSRRCI,
]


# RV32M Multiply/Divide Extension

MUL = Instruction("mul", InstructionFormat.R, InstructionCategory.MULDIV,
                  opcode=0b0110011, funct3=0b000, funct7=0b0000001,
                  reads_rs1=True, reads_rs2=True, writes_rd=True)
MULH = Instruction("mulh", InstructionFormat.R, InstructionCategory.MULDIV,
                   opcode=0b0110011, funct3=0b001, funct7=0b0000001,
                   reads_rs1=True, reads_rs2=True, writes_rd=True)
MULHSU = Instruction("mulhsu", InstructionFormat.R, InstructionCategory.MULDIV,
                     opcode=0b0110011, funct3=0b010, funct7=0b0000001,
                     reads_rs1=True, reads_rs2=True, writes_rd=True)
MULHU = Instruction("mulhu", InstructionFormat.R, InstructionCategory.MULDIV,
                    opcode=0b0110011, funct3=0b011, funct7=0b0000001,
                    reads_rs1=True, reads_rs2=True, writes_rd=True)
DIV = Instruction("div", InstructionFormat.R, InstructionCategory.MULDIV,
                  opcode=0b0110011, funct3=0b100, funct7=0b0000001,
                  reads_rs1=True, reads_rs2=True, writes_rd=True)
DIVU = Instruction("divu", InstructionFormat.R, InstructionCategory.MULDIV,
                   opcode=0b0110011, funct3=0b101, funct7=0b0000001,
                   reads_rs1=True, reads_rs2=True, writes_rd=True)
REM = Instruction("rem", InstructionFormat.R, InstructionCategory.MULDIV,
                  opcode=0b0110011, funct3=0b110, funct7=0b0000001,
                  reads_rs1=True, reads_rs2=True, writes_rd=True)
REMU = Instruction("remu", InstructionFormat.R, InstructionCategory.MULDIV,
                   opcode=0b0110011, funct3=0b111, funct7=0b0000001,
                   reads_rs1=True, reads_rs2=True, writes_rd=True)

RV32M_INSTRUCTIONS: List[Instruction] = [
    MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU,
]


def get_instructions_by_category(category: InstructionCategory,
                                 instructions: Optional[List[Instruction]] = None) -> List[Instruction]:
    """Get all instructions of a given category."""
    if instructions is None:
        instructions = RV32I_INSTRUCTIONS + RV32M_INSTRUCTIONS
    return [i for i in instructions if i.category == category]


def get_cf_ambiguous_instructions(instructions: Optional[List[Instruction]] = None) -> List[Instruction]:
    """Get all control-flow ambiguous instructions."""
    if instructions is None:
        instructions = RV32I_INSTRUCTIONS + RV32M_INSTRUCTIONS
    return [i for i in instructions if i.is_cf_ambiguous()]


def get_hopping_instructions(instructions: Optional[List[Instruction]] = None) -> List[Instruction]:
    """Get all hopping (control-flow changing) instructions."""
    if instructions is None:
        instructions = RV32I_INSTRUCTIONS + RV32M_INSTRUCTIONS
    return [i for i in instructions if i.is_hopping()]
