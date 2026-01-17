"""
RISC-V extension support.
"""

from typing import List, Set
from ..config import Extension
from .instructions import (
    Instruction, InstructionCategory,
    RV32I_INSTRUCTIONS, RV32M_INSTRUCTIONS,
    get_instructions_by_category,
)


def get_instructions_for_extensions(extensions: Set[Extension],
                                    xlen: int = 32) -> List[Instruction]:
    """Get list of instructions available for given extensions."""
    instructions = []

    if Extension.I in extensions:
        instructions.extend(RV32I_INSTRUCTIONS)

    if Extension.M in extensions:
        instructions.extend(RV32M_INSTRUCTIONS)

    # TODO: Add A, F, D, C extensions
    # if Extension.A in extensions:
    #     instructions.extend(RV32A_INSTRUCTIONS)
    # if Extension.F in extensions:
    #     instructions.extend(RV32F_INSTRUCTIONS)
    # if Extension.D in extensions:
    #     instructions.extend(RV32D_INSTRUCTIONS)
    # if Extension.C in extensions:
    #     instructions.extend(RVC_INSTRUCTIONS)

    # TODO: Add 64-bit instructions for xlen=64

    return instructions


def get_isa_string(extensions: Set[Extension], xlen: int = 32) -> str:
    """Generate ISA string (e.g., 'rv32imfd')."""
    prefix = f"rv{xlen}"

    ext_chars = []
    # Extensions must be in canonical order: IMAFDQC
    order = [Extension.I, Extension.M, Extension.A, Extension.F, Extension.D, Extension.C]
    for ext in order:
        if ext in extensions:
            ext_chars.append(ext.name.lower())

    return prefix + ''.join(ext_chars)


def filter_instructions_by_category(instructions: List[Instruction],
                                    categories: Set[InstructionCategory]) -> List[Instruction]:
    """Filter instructions by allowed categories."""
    return [i for i in instructions if i.category in categories]


def get_available_categories(extensions: Set[Extension]) -> Set[InstructionCategory]:
    """Get instruction categories available for given extensions."""
    categories = set()

    if Extension.I in extensions:
        categories.update([
            InstructionCategory.REGFSM,
            InstructionCategory.ALU,
            InstructionCategory.JAL,
            InstructionCategory.JALR,
            InstructionCategory.BRANCH,
            InstructionCategory.MEM,
            InstructionCategory.RDWRCSR,
            InstructionCategory.FENCES,
            InstructionCategory.EXCEPTION,
            InstructionCategory.DWNPRV,
        ])

    if Extension.M in extensions:
        categories.add(InstructionCategory.MULDIV)

    if Extension.F in extensions:
        categories.update([
            InstructionCategory.FPUFSM,
            InstructionCategory.MEMFPU,
            InstructionCategory.FPU,
        ])

    if Extension.D in extensions:
        categories.update([
            InstructionCategory.MEMFPUD,
            InstructionCategory.FPUD,
        ])

    return categories
