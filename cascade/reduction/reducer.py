"""
Main reduction orchestrator for Cascade.

Combines tail and head detection to produce minimal
bug-triggering programs.
"""

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from ..config import FuzzerConfig
from ..isa.instructions import JAL
from ..isa.encoding import EncodedInstruction
from ..generator.ultimate import UltimateProgram, ISSFeedback
from ..generator.basic_block import BasicBlock
from ..execution.rtl_runner import RTLRunner
from ..execution.iss_runner import ISSRunner
from ..execution.elf_writer import ELFWriter
from .tail_finder import TailFinder, TailResult
from .head_finder import HeadFinder, HeadResult


logger = logging.getLogger('cascade.reduction')


@dataclass
class ReductionResult:
    """Result of program reduction."""
    # Original program size (instructions)
    original_size: int

    # Reduced program size (instructions)
    reduced_size: int

    # Tail detection result
    tail: Optional[TailResult] = None

    # Head detection result
    head: Optional[HeadResult] = None

    # The reduced program
    reduced_program: Optional[UltimateProgram] = None

    # Total iterations for reduction
    total_iterations: int = 0

    @property
    def reduction_ratio(self) -> float:
        """Ratio of reduction (1.0 = no reduction)."""
        if self.original_size == 0:
            return 1.0
        return self.reduced_size / self.original_size


class Reducer:
    """
    Reduces bug-triggering programs to minimal size.

    Process:
    1. Find tail instruction (last instruction needed)
    2. Find head instruction (first instruction needed)
    3. Extract minimal program between head and tail
    """

    def __init__(self, config: FuzzerConfig,
                 rtl_runner: RTLRunner,
                 iss_runner: ISSRunner):
        """Initialize reducer."""
        self.config = config
        self.rtl_runner = rtl_runner
        self.iss_runner = iss_runner

        self.tail_finder = TailFinder(config, rtl_runner)
        self.head_finder = HeadFinder(config, rtl_runner, iss_runner)

        self.elf_writer = ELFWriter(config.cpu.xlen)

    def reduce(self, program: UltimateProgram,
               feedback: Optional[ISSFeedback] = None) -> ReductionResult:
        """
        Reduce a bug-triggering program.

        Args:
            program: The bug-triggering program
            feedback: Optional ISS feedback for state reconstruction

        Returns:
            ReductionResult with the reduced program
        """
        logger.info("Starting program reduction")

        result = ReductionResult(
            original_size=self._count_instructions(program)
        )

        # Verify the bug still triggers
        if not self._verify_bug(program):
            logger.error("Program no longer triggers bug, cannot reduce")
            result.reduced_program = program
            result.reduced_size = result.original_size
            return result

        # Step 1: Find tail
        logger.info("Finding tail instruction...")
        tail_result = self.tail_finder.find_tail(program)

        if tail_result is None:
            logger.warning("Could not find tail, returning original program")
            result.reduced_program = program
            result.reduced_size = result.original_size
            return result

        result.tail = tail_result
        result.total_iterations += tail_result.iterations
        logger.info(f"Found tail at PC 0x{tail_result.tail_pc:08x} "
                   f"(block {tail_result.tail_block_id}, "
                   f"instruction {tail_result.tail_instruction_index})")

        # Step 2: Find head
        logger.info("Finding head instruction...")
        head_result = self.head_finder.find_head(
            program, tail_result.tail_pc, feedback
        )

        if head_result is None:
            logger.warning("Could not find head, using tail-only reduction")
            result.reduced_program = self._create_tail_reduced_program(
                program, tail_result
            )
            result.reduced_size = self._count_instructions(result.reduced_program)
            return result

        result.head = head_result
        result.total_iterations += head_result.iterations
        logger.info(f"Found head at PC 0x{head_result.head_pc:08x} "
                   f"(block {head_result.head_block_id}, "
                   f"instruction {head_result.head_instruction_index})")

        # Step 3: Create minimal program
        logger.info("Creating minimal program...")
        result.reduced_program = self._create_minimal_program(
            program, head_result, tail_result, feedback
        )
        result.reduced_size = self._count_instructions(result.reduced_program)

        # Verify reduced program still triggers bug
        if not self._verify_bug(result.reduced_program):
            logger.warning("Reduced program doesn't trigger bug, "
                          "returning tail-only reduction")
            result.reduced_program = self._create_tail_reduced_program(
                program, tail_result
            )
            result.reduced_size = self._count_instructions(result.reduced_program)

        logger.info(f"Reduction complete: {result.original_size} -> "
                   f"{result.reduced_size} instructions "
                   f"({result.reduction_ratio:.1%})")

        return result

    def _verify_bug(self, program: UltimateProgram) -> bool:
        """Verify that the program triggers the bug."""
        result = self.rtl_runner.run(program)
        return result.bug_detected

    def _count_instructions(self, program: UltimateProgram) -> int:
        """Count total instructions in program."""
        return sum(b.num_instructions for b in program.blocks)

    def _create_tail_reduced_program(self, program: UltimateProgram,
                                      tail: TailResult) -> UltimateProgram:
        """
        Create a program reduced to just the tail.

        Keeps blocks 0 through tail_block, removing instructions
        after the tail.
        """
        reduced = UltimateProgram()
        reduced.entry_addr = program.entry_addr
        reduced.code_start = program.code_start
        reduced.data_start = program.data_start
        reduced.data_end = program.data_end

        for block in program.blocks:
            if block.block_id > tail.tail_block_id:
                break

            new_block = self._copy_block(block)

            if block.block_id == tail.tail_block_id:
                # Truncate at tail instruction
                idx = tail.tail_instruction_index
                if idx < len(new_block.instructions):
                    new_block.instructions = new_block.instructions[:idx + 1]
                    new_block.terminator = None
                # Add jump to final block
                new_block.terminator = EncodedInstruction(JAL, rd=0, imm=0)

            reduced.blocks.append(new_block)

        # Add final block (infinite loop)
        final_block = BasicBlock()
        final_block.block_id = tail.tail_block_id + 1
        final_block.start_addr = reduced.blocks[-1].end_addr if reduced.blocks else program.code_start
        final_block.terminator = EncodedInstruction(JAL, rd=0, imm=0)
        reduced.blocks.append(final_block)

        reduced.code_end = final_block.end_addr

        return reduced

    def _create_minimal_program(self, program: UltimateProgram,
                                head: HeadResult,
                                tail: TailResult,
                                feedback: Optional[ISSFeedback]) -> UltimateProgram:
        """
        Create a minimal program from head to tail.

        Includes context setter for state before head.
        """
        reduced = UltimateProgram()

        # Get state at head
        state = self.head_finder._get_state_at_block(
            program, head.head_block_id, feedback
        )

        if state is None:
            # Fall back to tail-only reduction
            return self._create_tail_reduced_program(program, tail)

        # Create context setter block
        context_block = self.head_finder._create_context_setter_block(state)
        context_block.start_addr = program.code_start
        reduced.blocks.append(context_block)

        # Copy blocks from head to tail
        in_range = False
        for block in program.blocks:
            if block.block_id == head.head_block_id:
                in_range = True
                # Copy block, potentially trimmed at head instruction
                new_block = self._copy_block(block)
                if head.head_instruction_index > 0:
                    new_block.instructions = new_block.instructions[head.head_instruction_index:]
                reduced.blocks.append(new_block)

            elif in_range and block.block_id <= tail.tail_block_id:
                new_block = self._copy_block(block)

                if block.block_id == tail.tail_block_id:
                    # Truncate at tail instruction
                    idx = tail.tail_instruction_index
                    if idx < len(new_block.instructions):
                        new_block.instructions = new_block.instructions[:idx + 1]
                        new_block.terminator = None
                    new_block.terminator = EncodedInstruction(JAL, rd=0, imm=0)
                    reduced.blocks.append(new_block)
                    break

                reduced.blocks.append(new_block)

        # Fix up context setter jump
        if len(reduced.blocks) > 1:
            target = reduced.blocks[1]
            jump_pc = context_block.end_addr - 4
            offset = target.start_addr - jump_pc
            context_block.terminator = EncodedInstruction(JAL, rd=0, imm=offset)

        # Add final block
        final_block = BasicBlock()
        final_block.block_id = tail.tail_block_id + 1
        if reduced.blocks:
            final_block.start_addr = reduced.blocks[-1].end_addr
        else:
            final_block.start_addr = program.code_start
        final_block.terminator = EncodedInstruction(JAL, rd=0, imm=0)
        reduced.blocks.append(final_block)

        reduced.entry_addr = context_block.start_addr
        reduced.code_start = context_block.start_addr
        reduced.code_end = final_block.end_addr
        reduced.data_start = program.data_start
        reduced.data_end = program.data_end

        return reduced

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

    def save_reduced(self, result: ReductionResult,
                     output_path: Path) -> None:
        """
        Save the reduced program to a file.

        Args:
            result: Reduction result
            output_path: Path to save the ELF file
        """
        if result.reduced_program is None:
            logger.error("No reduced program to save")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.elf_writer.write(result.reduced_program, output_path)
        logger.info(f"Saved reduced program to {output_path}")

        # Also save metadata
        meta_path = output_path.with_suffix('.txt')
        with open(meta_path, 'w') as f:
            f.write(f"Original size: {result.original_size} instructions\n")
            f.write(f"Reduced size: {result.reduced_size} instructions\n")
            f.write(f"Reduction ratio: {result.reduction_ratio:.1%}\n")
            f.write(f"Total iterations: {result.total_iterations}\n")

            if result.head:
                f.write(f"\nHead:\n")
                f.write(f"  Block: {result.head.head_block_id}\n")
                f.write(f"  Instruction: {result.head.head_instruction_index}\n")
                f.write(f"  PC: 0x{result.head.head_pc:08x}\n")

            if result.tail:
                f.write(f"\nTail:\n")
                f.write(f"  Block: {result.tail.tail_block_id}\n")
                f.write(f"  Instruction: {result.tail.tail_instruction_index}\n")
                f.write(f"  PC: 0x{result.tail.tail_pc:08x}\n")
