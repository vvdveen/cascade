"""
Head detection for program reduction.

Finds the first instruction involved in triggering the bug
using a context setter block and binary search.
"""

import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from ..config import FuzzerConfig
from ..isa.instructions import LUI, ADDI, JAL, CSRRW
from ..isa.encoding import EncodedInstruction
from ..generator.ultimate import UltimateProgram, ISSFeedback
from ..generator.basic_block import BasicBlock
from ..execution.rtl_runner import RTLRunner
from ..execution.iss_runner import ISSRunner


@dataclass
class HeadResult:
    """Result of head detection."""
    # Block containing the head instruction
    head_block_id: int

    # Index of head instruction within the block
    head_instruction_index: int

    # PC of the head instruction
    head_pc: int

    # Number of iterations to find
    iterations: int = 0


@dataclass
class ContextState:
    """Architectural state at a program point."""
    # Register values
    registers: Dict[int, int]

    # Program counter
    pc: int

    # Privilege level
    privilege: int = 3  # M-mode

    # CSR values (subset)
    csrs: Dict[int, int] = None

    def __post_init__(self):
        if self.csrs is None:
            self.csrs = {}


class HeadFinder:
    """
    Finds the head (first) instruction involved in triggering a bug.

    Uses a context setter block to restore architectural state
    at candidate head positions, then binary searches to find
    the minimal head.
    """

    def __init__(self, config: FuzzerConfig,
                 rtl_runner: RTLRunner,
                 iss_runner: ISSRunner):
        """Initialize head finder."""
        self.config = config
        self.rtl_runner = rtl_runner
        self.iss_runner = iss_runner

    def find_head(self, program: UltimateProgram,
                  tail_pc: int,
                  feedback: Optional[ISSFeedback] = None) -> Optional[HeadResult]:
        """
        Find the head instruction using binary search.

        Args:
            program: Bug-triggering program
            tail_pc: PC of the tail instruction (from TailFinder)
            feedback: ISS feedback for state reconstruction

        Returns:
            HeadResult if found, None otherwise
        """
        blocks = program.blocks

        if len(blocks) < 2:
            return None

        # Get the block containing the tail
        tail_block = None
        tail_block_idx = 0
        for i, block in enumerate(blocks):
            if block.start_addr <= tail_pc < block.end_addr:
                tail_block = block
                tail_block_idx = i
                break

        if tail_block is None:
            return None

        # Search from block 1 to tail_block (block 0 is setup)
        head_block, iterations = self._find_head_block(
            program, 1, tail_block_idx, feedback
        )

        if head_block is None:
            return None

        # Find specific instruction within block
        head_instr_idx, more_iters = self._find_head_instruction(
            program, head_block, feedback
        )
        iterations += more_iters

        if head_instr_idx is None:
            return None

        head_pc = head_block.start_addr + (head_instr_idx * 4)

        return HeadResult(
            head_block_id=head_block.block_id,
            head_instruction_index=head_instr_idx,
            head_pc=head_pc,
            iterations=iterations
        )

    def _find_head_block(self, program: UltimateProgram,
                         start_idx: int, end_idx: int,
                         feedback: Optional[ISSFeedback]) -> Tuple[Optional[BasicBlock], int]:
        """
        Find the basic block containing the head instruction.
        """
        blocks = program.blocks

        if start_idx >= end_idx:
            return blocks[start_idx] if start_idx < len(blocks) else None, 0

        low, high = start_idx, end_idx
        iterations = 0

        while low < high:
            mid = (low + high + 1) // 2  # Bias toward higher to find first block
            iterations += 1

            # Get state at beginning of block mid
            state = self._get_state_at_block(program, mid, feedback)

            if state is None:
                # Can't get state, assume bug is before mid
                high = mid - 1
                continue

            # Create program starting at mid with context setter
            modified = self._create_context_program(program, mid, state)

            result = self.rtl_runner.run(modified)

            if result.bug_detected:
                # Bug still triggers, head is at mid or later
                low = mid
            else:
                # Bug doesn't trigger, head is before mid
                high = mid - 1

        return blocks[low] if low < len(blocks) else None, iterations

    def _find_head_instruction(self, program: UltimateProgram,
                               block: BasicBlock,
                               feedback: Optional[ISSFeedback]) -> Tuple[Optional[int], int]:
        """
        Find the specific head instruction within a block.
        """
        n = block.num_instructions

        if n <= 1:
            return 0 if n == 1 else None, 0

        low, high = 0, n - 1
        iterations = 0

        while low < high:
            mid = (low + high + 1) // 2
            iterations += 1

            # Get state at instruction mid within block
            state = self._get_state_at_instruction(program, block, mid, feedback)

            if state is None:
                high = mid - 1
                continue

            # Create program starting at this instruction
            modified = self._create_instruction_context_program(
                program, block, mid, state
            )

            result = self.rtl_runner.run(modified)

            if result.bug_detected:
                low = mid
            else:
                high = mid - 1

        return low, iterations

    def _get_state_at_block(self, program: UltimateProgram,
                            block_idx: int,
                            feedback: Optional[ISSFeedback]) -> Optional[ContextState]:
        """
        Get architectural state at the beginning of a block.

        Uses ISS feedback if available, otherwise runs ISS.
        """
        blocks = program.blocks
        if block_idx >= len(blocks):
            return None

        block = blocks[block_idx]
        pc = block.start_addr

        if feedback and pc in feedback.register_values:
            return ContextState(
                registers=dict(feedback.register_values[pc]),
                pc=pc,
                privilege=feedback.privilege_levels.get(pc, 3)
            )

        # Run ISS to get state
        # For now, return a default state
        return ContextState(
            registers={i: 0 for i in range(32)},
            pc=pc,
            privilege=3
        )

    def _get_state_at_instruction(self, program: UltimateProgram,
                                  block: BasicBlock,
                                  instr_idx: int,
                                  feedback: Optional[ISSFeedback]) -> Optional[ContextState]:
        """
        Get state at a specific instruction within a block.
        """
        pc = block.start_addr + (instr_idx * 4)

        if feedback and pc in feedback.register_values:
            return ContextState(
                registers=dict(feedback.register_values[pc]),
                pc=pc
            )

        return ContextState(
            registers={i: 0 for i in range(32)},
            pc=pc
        )

    def _create_context_program(self, program: UltimateProgram,
                                start_block_idx: int,
                                state: ContextState) -> UltimateProgram:
        """
        Create a program with a context setter that jumps to the start block.

        The context setter loads all register values and then
        jumps to the start block.
        """
        modified = UltimateProgram()

        # Create context setter block
        context_block = self._create_context_setter_block(state)

        # Copy blocks from start_block_idx onwards
        modified.blocks = [context_block]
        for i in range(start_block_idx, len(program.blocks)):
            modified.blocks.append(self._copy_block(program.blocks[i]))

        # Fix up context setter to jump to first real block
        if len(modified.blocks) > 1:
            target = modified.blocks[1]
            jump_pc = context_block.end_addr - 4
            offset = target.start_addr - jump_pc
            context_block.terminator = EncodedInstruction(JAL, rd=0, imm=offset)

        modified.entry_addr = context_block.start_addr
        modified.code_start = context_block.start_addr
        modified.code_end = program.code_end
        modified.data_start = program.data_start
        modified.data_end = program.data_end

        return modified

    def _create_instruction_context_program(self, program: UltimateProgram,
                                            block: BasicBlock,
                                            start_instr_idx: int,
                                            state: ContextState) -> UltimateProgram:
        """
        Create a program that starts at a specific instruction.
        """
        # Similar to _create_context_program but modifies the target block
        modified = UltimateProgram()

        context_block = self._create_context_setter_block(state)
        modified.blocks = [context_block]

        # Find and copy blocks from target block onwards
        found = False
        for b in program.blocks:
            if b.block_id == block.block_id:
                found = True
                # Copy modified version of target block
                mod_block = self._copy_block(b)
                # Remove instructions before start_instr_idx
                mod_block.instructions = mod_block.instructions[start_instr_idx:]
                mod_block.start_addr = block.start_addr + (start_instr_idx * 4)
                modified.blocks.append(mod_block)
            elif found:
                modified.blocks.append(self._copy_block(b))

        # Fix up context setter
        if len(modified.blocks) > 1:
            target = modified.blocks[1]
            jump_pc = context_block.end_addr - 4
            offset = target.start_addr - jump_pc
            context_block.terminator = EncodedInstruction(JAL, rd=0, imm=offset)

        modified.entry_addr = context_block.start_addr
        modified.code_start = context_block.start_addr
        modified.code_end = program.code_end

        return modified

    def _create_context_setter_block(self, state: ContextState) -> BasicBlock:
        """
        Create a context setter block that loads register values.

        Uses lui/addi pairs to set each register.
        """
        block = BasicBlock()
        block.block_id = -1  # Special ID for context setter
        block.start_addr = self.config.memory.context_start

        for reg in range(1, 32):  # Skip x0
            value = state.registers.get(reg, 0)

            # lui rd, upper
            upper = (value + 0x800) >> 12
            block.instructions.append(EncodedInstruction(
                LUI, rd=reg, imm=(upper & 0xFFFFF) << 12
            ))

            # addi rd, rd, lower
            lower = value - (upper << 12)
            block.instructions.append(EncodedInstruction(
                ADDI, rd=reg, rs1=reg, imm=lower & 0xFFF
            ))

        # Terminator will be set by caller
        block.terminator = EncodedInstruction(JAL, rd=0, imm=0)

        return block

    def _copy_block(self, block: BasicBlock) -> BasicBlock:
        """Deep copy a basic block."""
        return BasicBlock(
            instructions=[copy.copy(i) for i in block.instructions],
            terminator=copy.copy(block.terminator) if block.terminator else None,
            start_addr=block.start_addr,
            block_id=block.block_id,
            cf_markers=list(block.cf_markers),
            fallthrough_addr=block.fallthrough_addr,
            jump_target_addr=block.jump_target_addr
        )
