"""
Register lifecycle state machine for Cascade.

Implements the register FSM from the paper (Figure 4) that tracks
register states for proper offset construction in cf-ambiguous instructions.

States:
- FREE: Available for any use
- GEN: Under generation (after lui r_off, imm1)
- READY: Offset register ready (after addi r_off, r_off, imm2)
- UNREL: Unreliable (after offset applied but not to this register)
- APPLIED: Applied register holding target value (after xor r_app, r_off, r_d)

Transitions:
FREE --(a)--> GEN --(b)--> READY --(c)--> APPLIED --(d)--> FREE
                              |
                              +--(c')--> UNREL --(e)--> FREE
                                            |
                                            +--(e')--> GEN
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple
import random


class RegisterState(Enum):
    """Register lifecycle states."""
    FREE = auto()      # Available for any use
    GEN = auto()       # Under generation (lui issued)
    READY = auto()     # Offset register ready
    UNREL = auto()     # Unreliable (offset used elsewhere)
    APPLIED = auto()   # Holding applied value


@dataclass
class RegisterInfo:
    """Information about a register's current state."""
    state: RegisterState = RegisterState.FREE

    # For GEN/READY states: the value being constructed
    pending_value: Optional[int] = None

    # For READY state: the dependent register (r_d) to XOR with
    dependent_reg: Optional[int] = None

    # For APPLIED state: the final value held
    applied_value: Optional[int] = None

    # Last instruction that wrote to this register
    last_write_pc: Optional[int] = None

    # Track write order for recency bias
    write_order: int = 0


class RegisterFSM:
    """
    Manages register state machine for all general-purpose registers.

    Key insight: We track register states to properly construct offsets
    for cf-ambiguous instructions. The XOR-based offset scheme allows
    us to entangle data flow with control flow.
    """

    def __init__(self, num_registers: int = 32):
        """Initialize register FSM."""
        self.num_registers = num_registers
        self.registers: Dict[int, RegisterInfo] = {}
        self.global_write_order = 0

        # Initialize all registers
        for i in range(num_registers):
            self.registers[i] = RegisterInfo()

        # x0 is always zero and never changes state
        self.registers[0].state = RegisterState.APPLIED
        self.registers[0].applied_value = 0

    def get_state(self, reg: int) -> RegisterState:
        """Get current state of a register."""
        return self.registers[reg].state

    def get_info(self, reg: int) -> RegisterInfo:
        """Get full info for a register."""
        return self.registers[reg]

    def get_free_registers(self) -> List[int]:
        """Get list of registers in FREE state."""
        return [r for r in range(1, self.num_registers)
                if self.registers[r].state == RegisterState.FREE]

    def get_ready_registers(self) -> List[int]:
        """Get list of registers in READY state."""
        return [r for r in range(1, self.num_registers)
                if self.registers[r].state == RegisterState.READY]

    def get_applied_registers(self) -> List[int]:
        """Get list of registers in APPLIED state (excluding x0)."""
        return [r for r in range(1, self.num_registers)
                if self.registers[r].state == RegisterState.APPLIED]

    def get_registers_with_known_value(self) -> List[int]:
        """Get registers that have a known value (APPLIED or READY)."""
        return [r for r in range(self.num_registers)
                if self.registers[r].state in (RegisterState.APPLIED, RegisterState.READY)]

    def get_recently_written(self, n: int = 5) -> List[int]:
        """Get n most recently written registers."""
        written = [(r, info.write_order)
                   for r, info in self.registers.items()
                   if r != 0 and info.last_write_pc is not None]
        written.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in written[:n]]

    def select_operand_register(self, bias_recent: float = 0.7) -> int:
        """
        Select a register for use as an operand.

        Uses recency bias: prefer recently-written registers with
        probability bias_recent.
        """
        recent = self.get_recently_written()
        if recent and random.random() < bias_recent:
            return random.choice(recent)

        # Fall back to any register with known value
        known = self.get_registers_with_known_value()
        if known:
            return random.choice(known)

        # Last resort: any non-x0 register
        return random.randint(1, self.num_registers - 1)

    # State transition methods

    def transition_lui(self, rd: int, imm: int, pc: int) -> None:
        """
        Transition (a): FREE -> GEN

        Called when lui rd, imm is issued to start constructing
        an offset value.
        """
        if rd == 0:
            return  # x0 is immutable

        self.registers[rd].state = RegisterState.GEN
        self.registers[rd].pending_value = imm << 12  # Upper 20 bits
        self.registers[rd].last_write_pc = pc
        self.global_write_order += 1
        self.registers[rd].write_order = self.global_write_order

    def transition_addi_complete(self, rd: int, imm: int, pc: int) -> None:
        """
        Transition (b): GEN -> READY

        Called when addi rd, rd, imm completes the offset construction.
        """
        if rd == 0:
            return

        info = self.registers[rd]
        if info.state != RegisterState.GEN:
            # If not in GEN state, treat as regular write
            self.mark_written(rd, pc)
            return

        # Complete the value construction
        upper = info.pending_value or 0
        # Sign-extend 12-bit immediate
        if imm & 0x800:
            imm = imm - 0x1000
        info.pending_value = (upper + imm) & 0xFFFFFFFF
        info.state = RegisterState.READY
        info.last_write_pc = pc
        self.global_write_order += 1
        info.write_order = self.global_write_order

    def transition_apply(self, rd: int, r_off: int, r_d: int,
                         result_value: int, pc: int) -> None:
        """
        Transition (c): READY -> APPLIED

        Called when xor rd, r_off, r_d applies the offset to get
        the final value.

        Also triggers (c'): READY -> UNREL for r_off if rd != r_off
        """
        if rd == 0:
            return

        # rd gets the applied value
        self.registers[rd].state = RegisterState.APPLIED
        self.registers[rd].applied_value = result_value
        self.registers[rd].last_write_pc = pc
        self.global_write_order += 1
        self.registers[rd].write_order = self.global_write_order

        # If r_off is different from rd and was READY, it becomes UNREL
        if r_off != rd and r_off != 0:
            if self.registers[r_off].state == RegisterState.READY:
                self.registers[r_off].state = RegisterState.UNREL

    def transition_free(self, reg: int) -> None:
        """
        Transition (d) or (e): -> FREE

        Called when a register's value is consumed and can be reused.
        """
        if reg == 0:
            return

        self.registers[reg].state = RegisterState.FREE
        self.registers[reg].pending_value = None
        self.registers[reg].dependent_reg = None
        self.registers[reg].applied_value = None

    def mark_written(self, rd: int, pc: int, value: Optional[int] = None) -> None:
        """
        Mark a register as written by a regular instruction.

        This is called for instructions that don't follow the
        lui/addi/xor pattern but still write to registers.
        """
        if rd == 0:
            return

        info = self.registers[rd]
        if value is not None:
            info.state = RegisterState.APPLIED
            info.applied_value = value
        else:
            info.state = RegisterState.FREE
            info.applied_value = None

        info.pending_value = None
        info.last_write_pc = pc
        self.global_write_order += 1
        info.write_order = self.global_write_order

    def begin_offset_construction(self, target_value: int) -> Optional[Tuple[int, int, int]]:
        """
        Begin constructing an offset to produce target_value.

        Returns (r_off, lui_imm, addi_imm) or None if no free register.

        The offset will be XOR'd with a dependent register later.
        """
        free = self.get_free_registers()
        if not free:
            return None

        r_off = random.choice(free)

        # Decompose target into lui + addi parts
        # Handle sign extension: if bit 11 is set, we need to add 1 to upper
        lui_imm = (target_value + 0x800) >> 12
        addi_imm = target_value - (lui_imm << 12)

        # Ensure addi_imm is in range [-2048, 2047]
        addi_imm = addi_imm & 0xFFF
        if addi_imm >= 0x800:
            addi_imm = addi_imm - 0x1000

        return (r_off, lui_imm & 0xFFFFF, addi_imm & 0xFFF)

    def compute_offset_for_value(self, target_value: int,
                                 r_d: int, r_d_value: int) -> int:
        """
        Compute offset such that: offset XOR r_d_value = target_value

        This is simply: offset = target_value XOR r_d_value
        """
        return target_value ^ r_d_value

    def reset(self) -> None:
        """Reset all register states to initial."""
        for i in range(self.num_registers):
            self.registers[i] = RegisterInfo()
        self.registers[0].state = RegisterState.APPLIED
        self.registers[0].applied_value = 0
        self.global_write_order = 0

    def set_initial_values(self, values: Dict[int, int]) -> None:
        """Set initial register values (e.g., from context setter)."""
        for reg, value in values.items():
            if reg == 0:
                continue
            self.registers[reg].state = RegisterState.APPLIED
            self.registers[reg].applied_value = value
