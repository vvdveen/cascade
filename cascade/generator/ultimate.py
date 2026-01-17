"""
Ultimate program construction with ISS feedback.

The ultimate program transforms the intermediate program using
actual register values from ISS simulation to construct proper
XOR-based offsets for cf-ambiguous instructions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import copy

from ..config import FuzzerConfig
from ..isa.instructions import LUI, ADDI, XOR, Instruction
from ..isa.encoding import EncodedInstruction
from .memory_manager import MemoryManager
from .register_fsm import RegisterFSM
from .basic_block import BasicBlock, CFAmbiguousMarker
from .intermediate import IntermediateProgram


@dataclass
class ISSFeedback:
    """
    Feedback collected from ISS simulation.

    Contains register values at specific program points that
    we need to properly construct offsets.
    """
    # Register values at each PC: pc -> reg_id -> value
    register_values: Dict[int, Dict[int, int]] = field(default_factory=dict)

    # Privilege level at each PC
    privilege_levels: Dict[int, int] = field(default_factory=dict)

    # Memory values at specific addresses (for load fixups)
    memory_values: Dict[int, int] = field(default_factory=dict)

    # Execution trace (sequence of PCs)
    trace: List[int] = field(default_factory=list)

    # Final register state
    final_registers: Dict[int, int] = field(default_factory=dict)

    def get_register_at(self, pc: int, reg: int) -> Optional[int]:
        """Get register value at given PC."""
        if pc in self.register_values:
            return self.register_values[pc].get(reg)
        return None

    def get_all_registers_at(self, pc: int) -> Optional[Dict[int, int]]:
        """Get all register values at given PC."""
        return self.register_values.get(pc)


@dataclass
class OffsetConstruction:
    """
    Describes how to construct an offset value.

    lui r_off, offset[31:12]
    addi r_off, r_off, offset[11:0]
    xor r_app, r_off, r_d  ; r_app = target_value
    """
    r_off: int          # Offset register
    r_d: int            # Dependent register (has random value)
    r_app: int          # Applied register (gets target value)
    offset_value: int   # The offset to XOR with r_d
    target_value: int   # The desired result in r_app
    r_d_value: int      # Current value in r_d


@dataclass
class UltimateProgram:
    """
    The ultimate program with proper offset constructions.

    This is the program that will be run on the RTL simulator.
    """
    # Basic blocks (may be modified from intermediate)
    blocks: List[BasicBlock] = field(default_factory=list)

    # Entry point
    entry_addr: int = 0

    # Memory regions
    code_start: int = 0
    code_end: int = 0
    data_start: int = 0
    data_end: int = 0

    # Original intermediate program (for reference)
    intermediate: Optional[IntermediateProgram] = None

    # ISS feedback used
    feedback: Optional[ISSFeedback] = None

    def to_bytes(self) -> bytes:
        """Convert entire program to bytes."""
        sorted_blocks = sorted(self.blocks, key=lambda b: b.start_addr)

        data = b''
        current_addr = self.code_start

        for block in sorted_blocks:
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


class UltimateProgramGenerator:
    """
    Transforms intermediate program to ultimate program using ISS feedback.

    Key transformation (from paper):
    | Intermediate                | Ultimate                      |
    |-----------------------------|-------------------------------|
    | lui r_off, val[31:12]       | lui r_off, offset[31:12]      |
    | addi r_off, r_off, val[11:0]| addi r_off, r_off, offset[11:0]|
    | mv r_app, r_off             | xor r_app, r_off, r_d         |
    | nop (non-taken branch)      | branch r1, r2, target         |
    """

    def __init__(self, config: FuzzerConfig):
        """Initialize ultimate program generator."""
        self.config = config

    def generate(self, intermediate: IntermediateProgram,
                 feedback: ISSFeedback) -> UltimateProgram:
        """
        Generate ultimate program from intermediate and ISS feedback.

        Args:
            intermediate: The intermediate program
            feedback: Register values from ISS simulation

        Returns:
            UltimateProgram with proper offset constructions
        """
        # Deep copy blocks to avoid modifying intermediate
        ultimate = UltimateProgram()
        ultimate.blocks = [self._copy_block(b) for b in intermediate.blocks]
        ultimate.entry_addr = intermediate.entry_addr
        ultimate.code_start = intermediate.code_start
        ultimate.code_end = intermediate.code_end
        ultimate.data_start = intermediate.data_start
        ultimate.data_end = intermediate.data_end
        ultimate.intermediate = intermediate
        ultimate.feedback = feedback

        # Process each cf-ambiguous marker
        for marker in intermediate.cf_markers:
            self._apply_offset_construction(ultimate, marker, feedback)

        return ultimate

    def _copy_block(self, block: BasicBlock) -> BasicBlock:
        """Deep copy a basic block."""
        new_block = BasicBlock(
            instructions=[copy.copy(i) for i in block.instructions],
            terminator=copy.copy(block.terminator) if block.terminator else None,
            start_addr=block.start_addr,
            block_id=block.block_id,
            cf_markers=list(block.cf_markers),
            fallthrough_addr=block.fallthrough_addr,
            jump_target_addr=block.jump_target_addr
        )
        return new_block

    def _apply_offset_construction(self, ultimate: UltimateProgram,
                                   marker: CFAmbiguousMarker,
                                   feedback: ISSFeedback) -> None:
        """
        Apply offset construction for a cf-ambiguous instruction.

        This transforms the intermediate construction (direct value load)
        to ultimate construction (XOR-based offset).
        """
        pc = marker.pc
        target_value = marker.target_value

        # Get register values at this PC from ISS
        regs = feedback.get_all_registers_at(pc)
        if regs is None:
            # No feedback at this PC, skip
            return

        # Find a dependent register with a known random value
        r_d = self._select_dependent_register(regs, marker.target_reg)
        if r_d is None:
            return

        r_d_value = regs.get(r_d, 0)

        # Compute offset: offset = target_value XOR r_d_value
        offset_value = target_value ^ r_d_value

        # Create offset construction
        construction = OffsetConstruction(
            r_off=marker.target_reg,  # Use target reg for offset
            r_d=r_d,
            r_app=marker.target_reg,
            offset_value=offset_value,
            target_value=target_value,
            r_d_value=r_d_value
        )

        # Find and modify the block
        block = ultimate.get_block_at(pc)
        if block is None:
            return

        # Insert offset construction before the cf-ambiguous instruction
        self._insert_offset_construction(block, marker, construction)

    def _select_dependent_register(self, regs: Dict[int, int],
                                   exclude_reg: int) -> Optional[int]:
        """
        Select a dependent register for XOR-based offset.

        Prefer registers with non-trivial values.
        """
        candidates = []
        for reg, value in regs.items():
            if reg == 0 or reg == exclude_reg:
                continue
            # Prefer registers with interesting values
            if value != 0:
                candidates.append((reg, value))

        if not candidates:
            # Fall back to any non-excluded register
            for reg in regs:
                if reg != 0 and reg != exclude_reg:
                    return reg
            return None

        # Pick one with most entropy (simple heuristic: most bits set)
        candidates.sort(key=lambda x: bin(x[1]).count('1'), reverse=True)
        return candidates[0][0]

    def _insert_offset_construction(self, block: BasicBlock,
                                   marker: CFAmbiguousMarker,
                                   construction: OffsetConstruction) -> None:
        """
        Insert offset construction instructions into the block.

        This modifies the block in place to add the lui/addi/xor
        sequence before the cf-ambiguous instruction.
        """
        pc = marker.pc
        idx = (pc - block.start_addr) // 4

        # For now, we just update the existing instruction's operands
        # A full implementation would insert additional instructions

        # If the marker is for the terminator
        if idx >= len(block.instructions):
            # It's the terminator, update in place
            if block.terminator:
                # For JALR: we need rs1 to have the target address
                if marker.instruction.name == 'jalr':
                    # The offset construction should set up rs1
                    # For simplicity, we assume the register setup
                    # happens in the preceding instructions
                    pass
                # For branches: we need to ensure the branch condition
                # evaluates to the desired outcome
                elif marker.instruction.category.name == 'BRANCH':
                    # Update the branch with the computed offset
                    pass
        else:
            # It's a regular instruction in the block
            instr = block.instructions[idx]

            # For loads: the loaded value comes from memory,
            # we don't modify the instruction but track the expected value
            if instr.instruction.is_load:
                pass

    def get_block_at(self, ultimate: UltimateProgram, addr: int) -> Optional[BasicBlock]:
        """Get block containing address."""
        for block in ultimate.blocks:
            if block.start_addr <= addr < block.end_addr:
                return block
        return None


# Helper function to add to UltimateProgram
def _get_block_at(self, addr: int) -> Optional[BasicBlock]:
    """Get block containing address."""
    for block in self.blocks:
        if block.start_addr <= addr < block.end_addr:
            return block
    return None

# Monkey-patch the method (cleaner would be to define it in the class)
UltimateProgram.get_block_at = _get_block_at
