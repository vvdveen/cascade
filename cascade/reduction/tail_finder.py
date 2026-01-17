"""
Tail detection for program reduction.

Finds the last instruction involved in triggering the bug
using binary search.
"""

import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List

from ..config import FuzzerConfig
from ..isa.instructions import JAL
from ..isa.encoding import EncodedInstruction
from ..generator.ultimate import UltimateProgram
from ..generator.basic_block import BasicBlock
from ..execution.rtl_runner import RTLRunner


@dataclass
class TailResult:
    """Result of tail detection."""
    # Block containing the tail instruction
    tail_block_id: int

    # Index of tail instruction within the block
    tail_instruction_index: int

    # PC of the tail instruction
    tail_pc: int

    # Number of iterations to find
    iterations: int = 0


class TailFinder:
    """
    Finds the tail (last) instruction involved in triggering a bug.

    Algorithm (binary search):
    1. Replace hopping instruction of basic block B_n-k with jump to final block
    2. If bug still triggers, bug is before B_n-k
    3. Binary search to find exact tail basic block, then tail instruction
    """

    def __init__(self, config: FuzzerConfig, rtl_runner: RTLRunner):
        """Initialize tail finder."""
        self.config = config
        self.rtl_runner = rtl_runner

    def find_tail(self, program: UltimateProgram) -> Optional[TailResult]:
        """
        Find the tail instruction using binary search.

        Args:
            program: Bug-triggering program

        Returns:
            TailResult if found, None otherwise
        """
        blocks = program.blocks

        if len(blocks) < 2:
            return None

        # First, find the tail block
        tail_block, iterations = self._find_tail_block(program)

        if tail_block is None:
            return None

        # Then, find the tail instruction within the block
        tail_instr_idx, more_iterations = self._find_tail_instruction(
            program, tail_block
        )
        iterations += more_iterations

        if tail_instr_idx is None:
            return None

        # Calculate PC
        tail_pc = tail_block.start_addr + (tail_instr_idx * 4)

        return TailResult(
            tail_block_id=tail_block.block_id,
            tail_instruction_index=tail_instr_idx,
            tail_pc=tail_pc,
            iterations=iterations
        )

    def _find_tail_block(self, program: UltimateProgram) -> Tuple[Optional[BasicBlock], int]:
        """
        Find the basic block containing the tail instruction.

        Uses binary search over blocks.
        """
        blocks = program.blocks
        n = len(blocks)

        if n < 2:
            return blocks[0] if blocks else None, 0

        # Find final block (the one with infinite loop)
        final_block = blocks[-1]

        # Binary search
        low, high = 0, n - 2  # Exclude final block
        iterations = 0

        while low < high:
            mid = (low + high) // 2
            iterations += 1

            # Create modified program with early exit at mid
            modified = self._create_early_exit_program(program, mid, final_block)

            # Test if bug still triggers
            result = self.rtl_runner.run(modified)

            if result.bug_detected:
                # Bug is in blocks 0..mid
                high = mid
            else:
                # Bug is in blocks mid+1..high
                low = mid + 1

        return blocks[low], iterations

    def _find_tail_instruction(self, program: UltimateProgram,
                              block: BasicBlock) -> Tuple[Optional[int], int]:
        """
        Find the specific instruction within the block.

        Uses binary search within the block.
        """
        n = block.num_instructions

        if n <= 1:
            return 0 if n == 1 else None, 0

        # Binary search within block
        low, high = 0, n - 1
        iterations = 0

        while low < high:
            mid = (low + high) // 2
            iterations += 1

            # Create modified program with early exit at instruction mid
            modified = self._create_instruction_exit_program(program, block, mid)

            result = self.rtl_runner.run(modified)

            if result.bug_detected:
                high = mid
            else:
                low = mid + 1

        return low, iterations

    def _create_early_exit_program(self, program: UltimateProgram,
                                   exit_block_idx: int,
                                   final_block: BasicBlock) -> UltimateProgram:
        """
        Create a modified program that exits early at given block.

        Replaces the terminator of block at exit_block_idx with
        a jump to the final block.
        """
        # Deep copy the program
        modified = UltimateProgram()
        modified.blocks = [self._copy_block(b) for b in program.blocks]
        modified.entry_addr = program.entry_addr
        modified.code_start = program.code_start
        modified.code_end = program.code_end
        modified.data_start = program.data_start
        modified.data_end = program.data_end

        # Modify the exit block's terminator
        exit_block = modified.blocks[exit_block_idx]
        term_pc = exit_block.end_addr - 4

        # Create jump to final block
        offset = final_block.start_addr - term_pc
        exit_block.terminator = EncodedInstruction(JAL, rd=0, imm=offset)

        return modified

    def _create_instruction_exit_program(self, program: UltimateProgram,
                                         block: BasicBlock,
                                         exit_instr_idx: int) -> UltimateProgram:
        """
        Create a modified program that exits after specific instruction.
        """
        modified = UltimateProgram()
        modified.blocks = [self._copy_block(b) for b in program.blocks]
        modified.entry_addr = program.entry_addr
        modified.code_start = program.code_start
        modified.code_end = program.code_end
        modified.data_start = program.data_start
        modified.data_end = program.data_end

        # Find the corresponding block in modified program
        target_block = None
        for b in modified.blocks:
            if b.block_id == block.block_id:
                target_block = b
                break

        if target_block is None:
            return modified

        # Find final block
        final_block = modified.blocks[-1]

        # Modify: convert instruction at exit_instr_idx to jump
        exit_pc = target_block.start_addr + (exit_instr_idx * 4)
        offset = final_block.start_addr - exit_pc

        if exit_instr_idx < len(target_block.instructions):
            target_block.instructions[exit_instr_idx] = EncodedInstruction(
                JAL, rd=0, imm=offset
            )
            # Remove subsequent instructions
            target_block.instructions = target_block.instructions[:exit_instr_idx + 1]
            target_block.terminator = None
        else:
            # It's the terminator
            target_block.terminator = EncodedInstruction(JAL, rd=0, imm=offset)

        return modified

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
