"""
Memory allocation and management for Cascade program generation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from ..config import MemoryRegion, MemoryLayout


@dataclass
class AllocatedBlock:
    """A block of allocated memory."""
    start: int
    size: int
    block_id: int
    is_code: bool = True

    @property
    def end(self) -> int:
        return self.start + self.size

    def contains(self, addr: int) -> bool:
        return self.start <= addr < self.end

    def overlaps(self, start: int, size: int) -> bool:
        return self.start < start + size and start < self.end


class MemoryManager:
    """
    Manages memory allocation for program generation.

    Responsibilities:
    - Progressive memory allocation for basic blocks
    - Strong allocation (forbid loads from specific areas)
    - Ensure no instruction overlap
    - Ensure stores don't overwrite instructions
    - Allocate space for context setter block (for reduction)
    """

    def __init__(self, layout: MemoryLayout, xlen: int = 32):
        """Initialize memory manager with layout configuration."""
        self.layout = layout
        self.xlen = xlen
        self.word_size = xlen // 8

        # Track allocated code blocks
        self.code_blocks: List[AllocatedBlock] = []
        self.next_block_id = 0

        # Current code allocation pointer
        self.code_ptr = layout.code_start

        # Track data regions for stores
        self.data_ptr = layout.data_start

        # Strong allocations: areas where loads are forbidden
        # This prevents loads from reading uninitialized or protected memory
        self.strong_allocations: List[MemoryRegion] = []

        # Store targets: pre-allocated regions for store instructions
        self.store_regions: List[MemoryRegion] = []

        # Reserve context setter area
        self.context_setter_reserved = MemoryRegion(
            start=layout.context_start,
            size=layout.context_size
        )

    def allocate_basic_block(self, num_instructions: int) -> AllocatedBlock:
        """
        Allocate memory for a basic block.

        Args:
            num_instructions: Number of instructions in the block

        Returns:
            AllocatedBlock with the allocated region
        """
        size = num_instructions * 4  # Each instruction is 4 bytes

        # Check if we have space
        if self.code_ptr + size > self.layout.code_start + self.layout.code_size:
            raise MemoryError("Code section exhausted")

        block = AllocatedBlock(
            start=self.code_ptr,
            size=size,
            block_id=self.next_block_id,
            is_code=True
        )

        self.code_ptr += size
        self.code_blocks.append(block)
        self.next_block_id += 1

        return block

    def allocate_data_region(self, size: int) -> MemoryRegion:
        """
        Allocate a data region for store instructions.

        Args:
            size: Size in bytes

        Returns:
            MemoryRegion for the allocated data
        """
        # Align to word boundary
        aligned_size = (size + self.word_size - 1) & ~(self.word_size - 1)

        if self.data_ptr + aligned_size > self.layout.data_start + self.layout.data_size:
            raise MemoryError("Data section exhausted")

        region = MemoryRegion(start=self.data_ptr, size=aligned_size)
        self.data_ptr += aligned_size
        self.store_regions.append(region)

        return region

    def add_strong_allocation(self, start: int, size: int) -> None:
        """
        Mark a region as strongly allocated (no loads allowed).

        This is used to prevent loads from reading uninitialized memory
        or protected regions.
        """
        self.strong_allocations.append(MemoryRegion(start=start, size=size))

    def can_load_from(self, addr: int, size: int = 4) -> bool:
        """
        Check if a load from the given address is valid.

        Valid load targets:
        - Code sections (read instructions as data)
        - Data sections that have been initialized
        - Not in strong allocation areas
        """
        # Check against strong allocations
        for region in self.strong_allocations:
            if region.contains(addr) or region.contains(addr + size - 1):
                return False

        # Must be in a valid memory region
        # Code region
        if (self.layout.code_start <= addr and
                addr + size <= self.layout.code_start + self.layout.code_size):
            # Only allow loads from already-allocated code
            for block in self.code_blocks:
                if block.contains(addr) and block.contains(addr + size - 1):
                    return True
            return False

        # Data region
        if (self.layout.data_start <= addr and
                addr + size <= self.layout.data_start + self.layout.data_size):
            return True

        return False

    def can_store_to(self, addr: int, size: int = 4) -> bool:
        """
        Check if a store to the given address is valid.

        Valid store targets:
        - Data sections only (never code)
        - Must not overflow section
        """
        # Only allow stores to data region
        if not (self.layout.data_start <= addr and
                addr + size <= self.layout.data_start + self.layout.data_size):
            return False

        # Don't allow stores that would overwrite code
        for block in self.code_blocks:
            if block.overlaps(addr, size):
                return False

        return True

    def get_valid_load_address(self, size: int, alignment: int = 4) -> Optional[int]:
        """
        Get a valid address for a load instruction.

        Returns a random valid address or None if no valid address exists.
        """
        # Try to find an address in allocated code blocks
        for block in self.code_blocks:
            # Align the start address
            aligned_start = (block.start + alignment - 1) & ~(alignment - 1)
            if aligned_start + size <= block.end:
                # Check not in strong allocation
                if self.can_load_from(aligned_start, size):
                    return aligned_start

        return None

    def get_valid_store_address(self, size: int, alignment: int = 4) -> Optional[int]:
        """
        Get a valid address for a store instruction.

        Returns a valid address in the data section.
        """
        # Use data region
        aligned_start = (self.layout.data_start + alignment - 1) & ~(alignment - 1)
        if aligned_start + size <= self.layout.data_start + self.layout.data_size:
            return aligned_start

        return None

    def get_stack_pointer(self) -> int:
        """Get initial stack pointer value."""
        return self.layout.stack_top

    def get_context_setter_region(self) -> MemoryRegion:
        """Get reserved region for context setter block."""
        return self.context_setter_reserved

    def get_code_bounds(self) -> Tuple[int, int]:
        """Get current code section bounds (start, end)."""
        return (self.layout.code_start, self.code_ptr)

    def get_data_bounds(self) -> Tuple[int, int]:
        """Get current data section bounds (start, end)."""
        return (self.layout.data_start, self.data_ptr)

    def reset(self) -> None:
        """Reset memory manager state."""
        self.code_blocks = []
        self.next_block_id = 0
        self.code_ptr = self.layout.code_start
        self.data_ptr = self.layout.data_start
        self.strong_allocations = []
        self.store_regions = []

    def get_block_for_address(self, addr: int) -> Optional[AllocatedBlock]:
        """Get the basic block containing the given address."""
        for block in self.code_blocks:
            if block.contains(addr):
                return block
        return None

    def instruction_index_in_block(self, addr: int, block: AllocatedBlock) -> int:
        """Get instruction index within a block."""
        if not block.contains(addr):
            raise ValueError("Address not in block")
        return (addr - block.start) // 4
