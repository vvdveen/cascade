"""
Tests for memory manager.
"""

import pytest
from cascade.config import MemoryLayout
from cascade.generator.memory_manager import MemoryManager, AllocatedBlock


class TestMemoryManager:
    """Test memory management."""

    @pytest.fixture
    def memory(self):
        """Create a memory manager with default layout."""
        layout = MemoryLayout()
        return MemoryManager(layout)

    def test_initial_state(self, memory):
        """Test initial memory manager state."""
        assert memory.code_ptr == memory.layout.code_start
        assert memory.data_ptr == memory.layout.data_start
        assert len(memory.code_blocks) == 0

    def test_allocate_basic_block(self, memory):
        """Test basic block allocation."""
        block = memory.allocate_basic_block(10)

        assert block.start == memory.layout.code_start
        assert block.size == 40  # 10 instructions * 4 bytes
        assert block.block_id == 0
        assert block.is_code

        # Code pointer should advance
        assert memory.code_ptr == memory.layout.code_start + 40

    def test_multiple_allocations(self, memory):
        """Test multiple block allocations."""
        block1 = memory.allocate_basic_block(5)
        block2 = memory.allocate_basic_block(10)

        assert block1.block_id == 0
        assert block2.block_id == 1
        assert block2.start == block1.end

    def test_allocation_overflow(self, memory):
        """Test allocation beyond code section."""
        # Try to allocate more than available
        huge_size = memory.layout.code_size // 4 + 1

        with pytest.raises(MemoryError):
            memory.allocate_basic_block(huge_size)

    def test_allocate_data_region(self, memory):
        """Test data region allocation."""
        region = memory.allocate_data_region(64)

        assert region.start == memory.layout.data_start
        assert region.size >= 64  # May be aligned up

    def test_can_load_from(self, memory):
        """Test load address validation."""
        # Allocate a code block
        block = memory.allocate_basic_block(10)

        # Can load from allocated code
        assert memory.can_load_from(block.start)
        assert memory.can_load_from(block.start + 4)

        # Cannot load from unallocated code
        assert not memory.can_load_from(block.end + 100)

    def test_can_store_to(self, memory):
        """Test store address validation."""
        # Cannot store to code region
        assert not memory.can_store_to(memory.layout.code_start)

        # Can store to data region
        assert memory.can_store_to(memory.layout.data_start)
        assert memory.can_store_to(memory.layout.data_start + 100)

    def test_strong_allocation(self, memory):
        """Test strong allocation (forbid loads)."""
        # Allocate a code block
        block = memory.allocate_basic_block(10)

        # Mark part of it as strongly allocated
        memory.add_strong_allocation(block.start, 8)

        # Cannot load from strongly allocated region
        assert not memory.can_load_from(block.start)
        assert not memory.can_load_from(block.start + 4)

        # Can still load from other parts
        assert memory.can_load_from(block.start + 12)

    def test_get_stack_pointer(self, memory):
        """Test stack pointer retrieval."""
        sp = memory.get_stack_pointer()
        assert sp == memory.layout.stack_top

    def test_reset(self, memory):
        """Test memory manager reset."""
        # Allocate some blocks
        memory.allocate_basic_block(10)
        memory.allocate_basic_block(10)
        memory.add_strong_allocation(0x80000000, 100)

        # Reset
        memory.reset()

        assert memory.code_ptr == memory.layout.code_start
        assert len(memory.code_blocks) == 0
        assert len(memory.strong_allocations) == 0


class TestAllocatedBlock:
    """Test AllocatedBlock helper class."""

    def test_end_property(self):
        """Test end address calculation."""
        block = AllocatedBlock(start=0x1000, size=100, block_id=0)
        assert block.end == 0x1064

    def test_contains(self):
        """Test address containment."""
        block = AllocatedBlock(start=0x1000, size=100, block_id=0)

        assert block.contains(0x1000)
        assert block.contains(0x1050)
        assert not block.contains(0x0FFF)
        assert not block.contains(0x1064)

    def test_overlaps(self):
        """Test overlap detection."""
        block = AllocatedBlock(start=0x1000, size=100, block_id=0)

        # Overlapping cases
        assert block.overlaps(0x1050, 100)  # Partial overlap
        assert block.overlaps(0x0F00, 0x200)  # Contains block
        assert block.overlaps(0x1010, 10)  # Inside block

        # Non-overlapping cases
        assert not block.overlaps(0x1064, 100)  # After block
        assert not block.overlaps(0x0E00, 0x200)  # Before block
