"""
Tests for register state machine.
"""

import pytest
from cascade.generator.register_fsm import RegisterFSM, RegisterState, RegisterInfo


class TestRegisterFSM:
    """Test register state machine."""

    @pytest.fixture
    def fsm(self):
        """Create a register FSM."""
        return RegisterFSM()

    def test_initial_state(self, fsm):
        """Test initial register states."""
        # All registers except x0 should be FREE
        for i in range(1, 32):
            assert fsm.get_state(i) == RegisterState.FREE

        # x0 is always APPLIED with value 0
        assert fsm.get_state(0) == RegisterState.APPLIED
        assert fsm.get_info(0).applied_value == 0

    def test_x0_immutable(self, fsm):
        """Test that x0 cannot be modified."""
        fsm.transition_lui(0, 0x12345, 0x1000)
        assert fsm.get_state(0) == RegisterState.APPLIED
        assert fsm.get_info(0).applied_value == 0

    def test_lui_transition(self, fsm):
        """Test LUI transition: FREE -> GEN."""
        fsm.transition_lui(1, 0x12345, 0x1000)

        assert fsm.get_state(1) == RegisterState.GEN
        info = fsm.get_info(1)
        assert info.pending_value == 0x12345000  # Upper 20 bits
        assert info.last_write_pc == 0x1000

    def test_addi_complete_transition(self, fsm):
        """Test ADDI transition: GEN -> READY."""
        # First do LUI
        fsm.transition_lui(1, 0x12345, 0x1000)

        # Then ADDI
        fsm.transition_addi_complete(1, 0x678, 0x1004)

        assert fsm.get_state(1) == RegisterState.READY
        info = fsm.get_info(1)
        # 0x12345000 + 0x678 = 0x12345678
        assert info.pending_value == 0x12345678

    def test_addi_with_negative_imm(self, fsm):
        """Test ADDI with negative immediate."""
        fsm.transition_lui(1, 0x12346, 0x1000)  # Need extra 1 for sign extension
        fsm.transition_addi_complete(1, 0xFFF, 0x1004)  # -1

        info = fsm.get_info(1)
        # 0x12346000 - 1 = 0x12345FFF
        assert info.pending_value == 0x12345FFF

    def test_apply_transition(self, fsm):
        """Test XOR apply transition: READY -> APPLIED."""
        # Set up offset register
        fsm.transition_lui(1, 0x12345, 0x1000)
        fsm.transition_addi_complete(1, 0x678, 0x1004)

        # Simulate a dependent register with some value
        fsm.mark_written(2, 0x0FFC, value=0xAAAAAAAA)

        # Apply: result = offset XOR dependent
        result = 0x12345678 ^ 0xAAAAAAAA
        fsm.transition_apply(3, r_off=1, r_d=2, result_value=result, pc=0x1008)

        assert fsm.get_state(3) == RegisterState.APPLIED
        assert fsm.get_info(3).applied_value == result

        # r_off should become UNREL since it's different from rd
        assert fsm.get_state(1) == RegisterState.UNREL

    def test_apply_same_register(self, fsm):
        """Test apply when rd == r_off."""
        fsm.transition_lui(1, 0x12345, 0x1000)
        fsm.transition_addi_complete(1, 0x678, 0x1004)
        fsm.mark_written(2, 0x0FFC, value=0xAAAAAAAA)

        result = 0x12345678 ^ 0xAAAAAAAA
        fsm.transition_apply(1, r_off=1, r_d=2, result_value=result, pc=0x1008)

        # Same register, should be APPLIED not UNREL
        assert fsm.get_state(1) == RegisterState.APPLIED

    def test_free_transition(self, fsm):
        """Test transition to FREE state."""
        fsm.mark_written(1, 0x1000, value=0x12345678)
        assert fsm.get_state(1) == RegisterState.APPLIED

        fsm.transition_free(1)
        assert fsm.get_state(1) == RegisterState.FREE

    def test_get_free_registers(self, fsm):
        """Test getting free registers."""
        free = fsm.get_free_registers()
        assert len(free) == 31  # All except x0

        # Mark some as used
        fsm.mark_written(1, 0x1000, value=0x100)
        fsm.mark_written(2, 0x1004, value=0x200)

        free = fsm.get_free_registers()
        assert 1 not in free
        assert 2 not in free

    def test_get_ready_registers(self, fsm):
        """Test getting ready registers."""
        assert len(fsm.get_ready_registers()) == 0

        fsm.transition_lui(1, 0x12345, 0x1000)
        fsm.transition_addi_complete(1, 0x678, 0x1004)

        ready = fsm.get_ready_registers()
        assert 1 in ready

    def test_get_recently_written(self, fsm):
        """Test getting recently written registers."""
        fsm.mark_written(5, 0x1000, value=0x100)
        fsm.mark_written(10, 0x1004, value=0x200)
        fsm.mark_written(15, 0x1008, value=0x300)

        recent = fsm.get_recently_written(2)
        assert len(recent) == 2
        assert 15 in recent  # Most recent
        assert 10 in recent  # Second most recent

    def test_compute_offset_for_value(self, fsm):
        """Test offset computation."""
        target = 0x12345678
        r_d_value = 0xAAAAAAAA

        offset = fsm.compute_offset_for_value(target, r_d=5, r_d_value=r_d_value)

        # offset XOR r_d_value = target
        assert offset ^ r_d_value == target

    def test_begin_offset_construction(self, fsm):
        """Test beginning offset construction."""
        result = fsm.begin_offset_construction(0x12345678)

        assert result is not None
        r_off, lui_imm, addi_imm = result

        # Verify the value can be reconstructed
        # Note: lui_imm is upper 20 bits, addi_imm is signed 12-bit
        value = (lui_imm << 12)
        if addi_imm >= 0x800:
            addi_imm = addi_imm - 0x1000
        value = (value + addi_imm) & 0xFFFFFFFF

        assert value == 0x12345678

    def test_reset(self, fsm):
        """Test FSM reset."""
        fsm.mark_written(1, 0x1000, value=0x100)
        fsm.mark_written(2, 0x1004, value=0x200)

        fsm.reset()

        for i in range(1, 32):
            assert fsm.get_state(i) == RegisterState.FREE

        assert fsm.get_state(0) == RegisterState.APPLIED
        assert fsm.global_write_order == 0
