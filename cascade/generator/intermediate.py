"""
Intermediate program construction for Cascade.

The intermediate program isolates control flow from data flow:
- CF-ambiguous instructions use placeholder values
- Non-taken branches become NOPs
- ISS simulation will reveal actual register values
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random

from ..config import FuzzerConfig
from ..isa.instructions import JAL, ADDI, LUI
from ..isa.encoding import EncodedInstruction, nop
from .memory_manager import MemoryManager
from .register_fsm import RegisterFSM
from .basic_block import BasicBlock, BasicBlockGenerator, CFAmbiguousMarker


@dataclass
class ProgramDescriptor:
    """
    Describes a generated program for reproducibility.

    Contains all information needed to reproduce the program.
    """
    seed: int
    num_blocks: int
    config_hash: str = ""


@dataclass
class IntermediateProgram:
    """
    An intermediate program with isolated control and data flow.

    The intermediate program can be run on ISS to determine
    actual register values at cf-ambiguous points.
    """
    # Basic blocks in program order
    blocks: List[BasicBlock] = field(default_factory=list)

    # All cf-ambiguous markers across all blocks
    cf_markers: List[CFAmbiguousMarker] = field(default_factory=list)

    # Program entry point
    entry_addr: int = 0

    # Memory regions
    code_start: int = 0
    code_end: int = 0
    data_start: int = 0
    data_end: int = 0

    # Descriptor for reproducibility
    descriptor: Optional[ProgramDescriptor] = None

    def get_block(self, block_id: int) -> Optional[BasicBlock]:
        """Get block by ID."""
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def get_block_at(self, addr: int) -> Optional[BasicBlock]:
        """Get block containing address."""
        for block in self.blocks:
            if block.start_addr <= addr < block.end_addr:
                return block
        return None

    def get_instruction_at(self, pc: int) -> Optional[EncodedInstruction]:
        """Get instruction at given PC."""
        block = self.get_block_at(pc)
        if block:
            return block.get_instruction_at(pc)
        return None

    def to_bytes(self) -> bytes:
        """Convert entire program to bytes."""
        # Sort blocks by address
        sorted_blocks = sorted(self.blocks, key=lambda b: b.start_addr)

        data = b''
        current_addr = self.code_start

        for block in sorted_blocks:
            # Pad if there's a gap
            if block.start_addr > current_addr:
                gap = block.start_addr - current_addr
                data += b'\x00' * gap

            data += block.to_bytes()
            current_addr = block.end_addr

        return data

    @property
    def size(self) -> int:
        """Total size of program in bytes."""
        if not self.blocks:
            return 0
        return self.code_end - self.code_start


class IntermediateProgramGenerator:
    """
    Generates intermediate programs for ISS simulation.

    Key insight: In the intermediate program, control flow is
    deterministic because we set registers directly to target
    values (no XOR offset construction).
    """

    def __init__(self, config: FuzzerConfig):
        """Initialize intermediate program generator."""
        self.config = config
        self.memory = MemoryManager(config.memory, config.cpu.xlen)
        self.reg_fsm = RegisterFSM(config.cpu.num_gpr)
        self.block_gen = BasicBlockGenerator(config, self.memory, self.reg_fsm)

    def generate(self, seed: Optional[int] = None) -> IntermediateProgram:
        """
        Generate an intermediate program.

        Args:
            seed: Random seed for reproducibility

        Returns:
            IntermediateProgram ready for ISS simulation
        """
        if seed is not None:
            random.seed(seed)
        else:
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)

        # Reset state
        self.memory.reset()
        self.reg_fsm.reset()

        program = IntermediateProgram()
        program.entry_addr = self.memory.layout.code_start
        program.code_start = self.memory.layout.code_start
        program.data_start = self.memory.layout.data_start

        # Generate initial block
        initial_block = self._generate_initial_block()
        program.blocks.append(initial_block)

        # Generate fuzzing blocks
        num_blocks = random.randint(
            self.config.min_basic_blocks,
            self.config.max_basic_blocks
        )

        for i in range(num_blocks):
            block = self._generate_fuzzing_block(i + 1)
            program.blocks.append(block)

        # Generate final block
        final_block = self._generate_final_block(num_blocks + 1)
        program.blocks.append(final_block)

        # Fix up control flow targets
        self._fixup_control_flow(program)

        # Collect all cf-ambiguous markers
        for block in program.blocks:
            program.cf_markers.extend(block.cf_markers)

        # Set bounds
        program.code_end = self.memory.code_ptr
        program.data_end = self.memory.data_ptr

        # Create descriptor
        program.descriptor = ProgramDescriptor(
            seed=seed,
            num_blocks=len(program.blocks)
        )

        return program

    def _generate_initial_block(self) -> BasicBlock:
        """Generate the initial setup block."""
        # Allocate space
        block_size = 50  # Enough for setup
        alloc = self.memory.allocate_basic_block(block_size)

        block = self.block_gen.generate_initial_block(alloc.start)
        return block

    def _generate_fuzzing_block(self, block_id: int) -> BasicBlock:
        """Generate a fuzzing block."""
        # Random number of instructions
        num_instrs = random.randint(
            self.config.min_block_instructions,
            self.config.max_block_instructions
        )

        # Allocate space (include terminator)
        alloc = self.memory.allocate_basic_block(num_instrs + 1)

        block = self.block_gen.generate_block(
            alloc.start,
            block_id,
            min_instrs=num_instrs,
            max_instrs=num_instrs
        )

        return block

    def _generate_final_block(self, block_id: int) -> BasicBlock:
        """Generate the final completion block."""
        block_size = 3 if self.config.cpu.name == "kronos" else 1
        alloc = self.memory.allocate_basic_block(block_size)
        block = self.block_gen.generate_final_block(alloc.start, block_id)
        return block

    def _fixup_control_flow(self, program: IntermediateProgram) -> None:
        """
        Fix up control flow targets in the program.

        For the intermediate program, all control flow is made
        deterministic by ensuring targets are reachable.
        """
        blocks = program.blocks

        for i, block in enumerate(blocks):
            if block.terminator is None:
                continue

            term = block.terminator
            term_pc = block.end_addr - 4

            if term.instruction.name == 'jal':
                # Jump to next block
                if i + 1 < len(blocks):
                    next_block = blocks[i + 1]
                    offset = next_block.start_addr - term_pc
                    block.terminator = EncodedInstruction(
                        JAL,
                        rd=term.rd,
                        imm=offset
                    )
                    block.jump_target_addr = next_block.start_addr
                else:
                    # Jump to self (end)
                    block.terminator = EncodedInstruction(JAL, rd=0, imm=0)

            elif term.instruction.name == 'jalr':
                # For intermediate program, make JALR jump to next block
                # by setting up the register appropriately
                if i + 1 < len(blocks):
                    next_block = blocks[i + 1]
                    block.jump_target_addr = next_block.start_addr

                    # Update the marker with target value
                    for marker in block.cf_markers:
                        if marker.pc == term_pc:
                            marker.target_value = next_block.start_addr
                            marker.branch_target = next_block.start_addr
                    # Overwrite tail instructions to set rs1 to target
                    setup = self._encode_li(term.rs1, next_block.start_addr)
                    if not self._overwrite_tail_instructions(block, setup):
                        # Fallback: replace with JAL if we can't set up registers
                        offset = next_block.start_addr - term_pc
                        block.terminator = EncodedInstruction(JAL, rd=term.rd, imm=offset)

            elif term.instruction.category.name == 'BRANCH':
                # For intermediate program, decide if branch is taken
                # and set up registers accordingly
                if i + 1 < len(blocks):
                    next_block = blocks[i + 1]
                    block.fallthrough_addr = next_block.start_addr

                    # Find the marker for this branch
                    for marker in block.cf_markers:
                        if marker.pc == term_pc:
                            if marker.is_taken:
                                # Pick a random future block as target
                                future_blocks = blocks[i + 2:] if i + 2 < len(blocks) else []
                                candidates = [
                                    b for b in future_blocks
                                    if self._branch_offset_ok(b.start_addr - term_pc)
                                ]
                                if candidates:
                                    target = random.choice(candidates)
                                else:
                                    target = next_block
                                offset = target.start_addr - term_pc
                                if not self._branch_offset_ok(offset):
                                    block.terminator = EncodedInstruction(JAL, rd=0, imm=offset)
                                    block.jump_target_addr = target.start_addr
                                else:
                                    block.terminator = EncodedInstruction(
                                        term.instruction,
                                        rs1=term.rs1,
                                        rs2=term.rs2,
                                        imm=offset
                                    )
                                    marker.branch_target = target.start_addr
                                    block.jump_target_addr = target.start_addr
                            else:
                                # Branch not taken, just fall through
                                offset = next_block.start_addr - term_pc
                                if not self._branch_offset_ok(offset):
                                    block.terminator = EncodedInstruction(JAL, rd=0, imm=offset)
                                    block.jump_target_addr = next_block.start_addr
                                else:
                                    block.terminator = EncodedInstruction(
                                        term.instruction,
                                        rs1=term.rs1,
                                        rs2=term.rs2,
                                        imm=offset
                                    )
                                    marker.branch_target = next_block.start_addr

                            # Overwrite tail instructions to force branch outcome
                            rs1_val, rs2_val = self._branch_setup_values(term.instruction.name, marker.is_taken)
                            setup = self._encode_li(term.rs1, rs1_val) + self._encode_li(term.rs2, rs2_val)
                            if not self._overwrite_tail_instructions(block, setup):
                                # Fallback: replace with JAL to the chosen target
                                target_addr = marker.branch_target or next_block.start_addr
                                offset = target_addr - term_pc
                                block.terminator = EncodedInstruction(JAL, rd=0, imm=offset)
                                block.jump_target_addr = target_addr

    def reset(self) -> None:
        """Reset generator state."""
        self.memory.reset()
        self.reg_fsm.reset()

    def _encode_li(self, reg: int, value: int) -> List[EncodedInstruction]:
        """Encode a 32-bit immediate load into register."""
        upper = (value + 0x800) >> 12
        lower = value - (upper << 12)
        return [
            EncodedInstruction(LUI, rd=reg, imm=(upper & 0xFFFFF) << 12),
            EncodedInstruction(ADDI, rd=reg, rs1=reg, imm=lower & 0xFFF),
        ]

    def _branch_offset_ok(self, offset: int) -> bool:
        """Check if branch offset fits 13-bit signed immediate (and is aligned)."""
        if offset % 2 != 0:
            return False
        return -4094 <= offset <= 4094

    def _overwrite_tail_instructions(self, block: BasicBlock,
                                     setup: List[EncodedInstruction]) -> bool:
        """Overwrite the tail of block.instructions with setup instructions."""
        if len(block.instructions) < len(setup):
            return False
        start = len(block.instructions) - len(setup)
        block.instructions[start:] = setup
        return True

    def _branch_setup_values(self, instr_name: str, is_taken: bool) -> Tuple[int, int]:
        """Pick register values that force a branch to be taken or not."""
        if instr_name == "beq":
            return (0, 0) if is_taken else (0, 1)
        if instr_name == "bne":
            return (0, 1) if is_taken else (0, 0)
        if instr_name in ("blt", "bltu"):
            return (0, 1) if is_taken else (1, 0)
        if instr_name in ("bge", "bgeu"):
            return (1, 0) if is_taken else (0, 1)
        return (0, 0)
