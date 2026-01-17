"""
Tests for program generation.
"""

import pytest
from cascade.config import FuzzerConfig, CPUConfig, Extension
from cascade.generator.intermediate import IntermediateProgram, IntermediateProgramGenerator
from cascade.generator.basic_block import BasicBlock


class TestIntermediateProgramGeneration:
    """Test intermediate program generation."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return FuzzerConfig(
            cpu=CPUConfig(
                name="test",
                xlen=32,
                extensions={Extension.I, Extension.M}
            ),
            min_basic_blocks=5,
            max_basic_blocks=10,
            min_block_instructions=2,
            max_block_instructions=5,
        )

    @pytest.fixture
    def generator(self, config):
        """Create a program generator."""
        return IntermediateProgramGenerator(config)

    def test_generate_program(self, generator):
        """Test basic program generation."""
        program = generator.generate(seed=42)

        assert isinstance(program, IntermediateProgram)
        assert len(program.blocks) > 0

    def test_program_has_entry_point(self, generator):
        """Test that program has valid entry point."""
        program = generator.generate(seed=42)

        assert program.entry_addr == program.code_start
        assert program.entry_addr > 0

    def test_program_has_initial_block(self, generator):
        """Test that program has initial setup block."""
        program = generator.generate(seed=42)

        initial_block = program.blocks[0]
        assert initial_block.block_id == 0
        assert initial_block.start_addr == program.entry_addr

    def test_program_has_final_block(self, generator):
        """Test that program has final block."""
        program = generator.generate(seed=42)

        final_block = program.blocks[-1]
        # Final block should have terminator (infinite loop)
        assert final_block.terminator is not None

    def test_blocks_are_contiguous(self, generator):
        """Test that blocks are laid out contiguously."""
        program = generator.generate(seed=42)

        for i in range(len(program.blocks) - 1):
            curr = program.blocks[i]
            next_ = program.blocks[i + 1]
            # Next block should start at or after current block ends
            assert next_.start_addr >= curr.end_addr

    def test_program_to_bytes(self, generator):
        """Test program serialization to bytes."""
        program = generator.generate(seed=42)

        data = program.to_bytes()

        assert isinstance(data, bytes)
        assert len(data) > 0
        # Should be multiple of 4 (instruction size)
        assert len(data) % 4 == 0

    def test_reproducibility(self, generator):
        """Test that same seed produces same program."""
        program1 = generator.generate(seed=12345)
        generator.reset()
        program2 = generator.generate(seed=12345)

        assert program1.to_bytes() == program2.to_bytes()

    def test_different_seeds_different_programs(self, generator):
        """Test that different seeds produce different programs."""
        program1 = generator.generate(seed=1)
        generator.reset()
        program2 = generator.generate(seed=2)

        # Programs should be different (with high probability)
        assert program1.to_bytes() != program2.to_bytes()

    def test_block_size_constraints(self, generator, config):
        """Test that block sizes respect constraints."""
        program = generator.generate(seed=42)

        # Skip initial and final blocks which have special sizes
        for block in program.blocks[1:-1]:
            num_instrs = block.num_instructions
            # Should be within bounds (including terminator)
            assert num_instrs >= 1
            assert num_instrs <= config.max_block_instructions + 1

    def test_program_descriptor(self, generator):
        """Test that program has descriptor for reproducibility."""
        program = generator.generate(seed=42)

        assert program.descriptor is not None
        assert program.descriptor.seed == 42
        assert program.descriptor.num_blocks == len(program.blocks)


class TestBasicBlock:
    """Test BasicBlock class."""

    def test_empty_block(self):
        """Test empty block properties."""
        block = BasicBlock()

        assert block.num_instructions == 0
        assert block.size == 0
        assert block.to_bytes() == b''

    def test_block_with_instructions(self):
        """Test block with instructions."""
        from cascade.isa.encoding import nop, jal

        block = BasicBlock()
        block.instructions = [nop(), nop()]
        block.terminator = jal(0, 4)

        assert block.num_instructions == 3
        assert block.size == 12  # 3 instructions * 4 bytes

    def test_block_to_bytes(self):
        """Test block serialization."""
        from cascade.isa.encoding import nop

        block = BasicBlock()
        block.instructions = [nop()]
        block.terminator = nop()

        data = block.to_bytes()
        assert len(data) == 8  # 2 instructions

    def test_get_instruction_at(self):
        """Test instruction retrieval by PC."""
        from cascade.isa.encoding import nop, addi

        block = BasicBlock(start_addr=0x1000)
        block.instructions = [nop(), addi(1, 0, 100)]
        block.terminator = nop()

        # Get first instruction
        instr = block.get_instruction_at(0x1000)
        assert instr is not None
        assert instr.instruction.name == 'addi'
        assert instr.rd == 0  # NOP is addi x0, x0, 0

        # Get second instruction
        instr = block.get_instruction_at(0x1004)
        assert instr is not None
        assert instr.rd == 1

        # Get terminator
        instr = block.get_instruction_at(0x1008)
        assert instr is not None

        # Out of bounds
        assert block.get_instruction_at(0x0FFC) is None
        assert block.get_instruction_at(0x100C) is None
