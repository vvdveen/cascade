"""
Basic block generation for Cascade.

A basic block consists of:
- 0 or more non-control-flow (still) instructions
- 1 control-flow (hopping) instruction at the end
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
import random

from ..config import FuzzerConfig, InstructionWeights
from ..isa.instructions import (
    Instruction, InstructionCategory, ControlFlowType,
    RV32I_INSTRUCTIONS, RV32M_INSTRUCTIONS,
    get_instructions_by_category, get_hopping_instructions,
    LUI, ADDI, ADD, XOR, JAL, JALR, BEQ, BNE, BLT, BGE, BLTU, BGEU,
    LW, SW, LB, LH, LBU, LHU, SB, SH,
    EBREAK,
)
from ..isa.encoding import EncodedInstruction
from .memory_manager import MemoryManager
from .register_fsm import RegisterFSM, RegisterState


@dataclass
class CFAmbiguousMarker:
    """
    Marker for a cf-ambiguous instruction.

    In the intermediate program, we record where cf-ambiguous instructions
    are located so we can later fix them up with proper offsets.
    """
    pc: int                      # Program counter of the instruction
    instruction: Instruction     # The instruction
    encoded: EncodedInstruction  # The encoded instruction
    target_reg: int             # Register that needs specific value
    target_value: int           # Value needed in target_reg
    is_taken: bool              # For branches: should it be taken?
    branch_target: Optional[int] = None  # For branches: target address


@dataclass
class BasicBlock:
    """
    A basic block of instructions.

    Structure:
    - instructions: List of encoded instructions
    - terminator: The hopping instruction at the end
    - start_addr: Starting address of the block
    - markers: CF-ambiguous instruction markers for later fixup
    """
    instructions: List[EncodedInstruction] = field(default_factory=list)
    terminator: Optional[EncodedInstruction] = None
    start_addr: int = 0
    block_id: int = 0

    # Markers for cf-ambiguous instructions
    cf_markers: List[CFAmbiguousMarker] = field(default_factory=list)

    # Next block addresses (for control flow)
    fallthrough_addr: Optional[int] = None
    jump_target_addr: Optional[int] = None

    @property
    def size(self) -> int:
        """Size of block in bytes."""
        n = len(self.instructions)
        if self.terminator:
            n += 1
        return n * 4

    @property
    def end_addr(self) -> int:
        """Address after the last instruction."""
        return self.start_addr + self.size

    @property
    def num_instructions(self) -> int:
        """Total number of instructions including terminator."""
        n = len(self.instructions)
        if self.terminator:
            n += 1
        return n

    def get_instruction_at(self, pc: int) -> Optional[EncodedInstruction]:
        """Get instruction at given PC."""
        if pc < self.start_addr or pc >= self.end_addr:
            return None
        idx = (pc - self.start_addr) // 4
        if idx < len(self.instructions):
            return self.instructions[idx]
        elif self.terminator and idx == len(self.instructions):
            return self.terminator
        return None

    def to_bytes(self) -> bytes:
        """Convert block to bytes."""
        data = b''
        for instr in self.instructions:
            data += instr.to_bytes()
        if self.terminator:
            data += self.terminator.to_bytes()
        return data


class BasicBlockGenerator:
    """
    Generates basic blocks for the intermediate program.
    """

    def __init__(self, config: FuzzerConfig, memory: MemoryManager,
                 reg_fsm: RegisterFSM):
        """Initialize basic block generator."""
        self.config = config
        self.memory = memory
        self.reg_fsm = reg_fsm
        self.data_base_reg = 18  # Reserved register for data base
        self.reserved_regs = {self.data_base_reg}

        # Build instruction pools by category
        self.instruction_pools = self._build_instruction_pools()

        # Get normalized weights
        self.weights = config.weights.normalize()

    def _build_instruction_pools(self) -> Dict[str, List[Instruction]]:
        """Build instruction pools organized by category."""
        from ..isa.extensions import get_instructions_for_extensions

        available = get_instructions_for_extensions(
            self.config.cpu.extensions,
            self.config.cpu.xlen
        )

        pools = {}
        for cat_name in ['alu', 'mem', 'branch', 'jal', 'jalr', 'regfsm',
                         'csr', 'fence', 'muldiv', 'exception', 'privilege']:
            pools[cat_name] = []

        for instr in available:
            if instr.category == InstructionCategory.ALU:
                pools['alu'].append(instr)
            elif instr.category == InstructionCategory.MEM:
                pools['mem'].append(instr)
            elif instr.category == InstructionCategory.BRANCH:
                pools['branch'].append(instr)
            elif instr.category == InstructionCategory.JAL:
                pools['jal'].append(instr)
            elif instr.category == InstructionCategory.JALR:
                pools['jalr'].append(instr)
            elif instr.category == InstructionCategory.REGFSM:
                pools['regfsm'].append(instr)
            elif instr.category == InstructionCategory.RDWRCSR:
                pools['csr'].append(instr)
            elif instr.category == InstructionCategory.FENCES:
                pools['fence'].append(instr)
            elif instr.category == InstructionCategory.MULDIV:
                pools['muldiv'].append(instr)
            elif instr.category == InstructionCategory.EXCEPTION:
                pools['exception'].append(instr)
            elif instr.category == InstructionCategory.DWNPRV:
                pools['privilege'].append(instr)

        return pools

    def generate_block(self, start_addr: int, block_id: int,
                       min_instrs: int = 1,
                       max_instrs: int = 20) -> BasicBlock:
        """
        Generate a complete basic block.

        Args:
            start_addr: Starting address of the block
            block_id: Unique block identifier
            min_instrs: Minimum number of still instructions
            max_instrs: Maximum number of still instructions

        Returns:
            Generated BasicBlock
        """
        block = BasicBlock(start_addr=start_addr, block_id=block_id)

        # Generate 0 to N still instructions
        num_still = random.randint(min_instrs, max_instrs)
        pc = start_addr

        for _ in range(num_still):
            instr, marker = self._generate_still_instruction(pc)
            block.instructions.append(instr)
            if marker:
                block.cf_markers.append(marker)
            pc += 4

        # Generate terminating hopping instruction
        terminator, marker = self._generate_hopping_instruction(pc)
        block.terminator = terminator
        if marker:
            block.cf_markers.append(marker)

        return block

    def _select_category(self, allow_hopping: bool = False) -> str:
        """Select instruction category based on weights."""
        if allow_hopping:
            categories = list(self.weights.keys())
        else:
            # Exclude hopping categories for still instructions
            categories = [c for c in self.weights.keys()
                          if c not in ('branch', 'jal', 'jalr', 'exception', 'privilege')]

        total = sum(self.weights.get(c, 0) for c in categories)
        if total == 0:
            return 'alu'

        r = random.random() * total
        cumulative = 0
        for cat in categories:
            cumulative += self.weights.get(cat, 0)
            if r <= cumulative:
                return cat

        return categories[-1] if categories else 'alu'

    def _generate_still_instruction(self, pc: int) -> Tuple[EncodedInstruction,
                                                            Optional[CFAmbiguousMarker]]:
        """Generate a non-control-flow instruction."""
        category = self._select_category(allow_hopping=False)

        if category == 'alu':
            return self._generate_alu_instruction(pc), None
        elif category == 'mem':
            return self._generate_mem_instruction(pc)
        elif category == 'regfsm':
            return self._generate_regfsm_instruction(pc), None
        elif category == 'muldiv':
            return self._generate_muldiv_instruction(pc), None
        else:
            # Default to ALU
            return self._generate_alu_instruction(pc), None

    def _generate_hopping_instruction(self, pc: int) -> Tuple[EncodedInstruction,
                                                               Optional[CFAmbiguousMarker]]:
        """Generate a control-flow changing instruction."""
        # Choose between JAL, JALR, and BRANCH
        choice = random.choice(['jal', 'branch', 'jalr'])

        if choice == 'jal':
            return self._generate_jal_instruction(pc)
        elif choice == 'jalr':
            return self._generate_jalr_instruction(pc)
        else:
            return self._generate_branch_instruction(pc)

    def _generate_alu_instruction(self, pc: int) -> EncodedInstruction:
        """Generate an ALU instruction."""
        alu_instrs = self.instruction_pools.get('alu', [])
        if not alu_instrs:
            # Fallback to ADD
            instr = ADD
        else:
            instr = random.choice(alu_instrs)

        rd = self._select_dest_register()
        rs1 = self.reg_fsm.select_operand_register(self.config.recent_register_bias)
        rs2 = self.reg_fsm.select_operand_register(self.config.recent_register_bias)

        # For I-type ALU instructions
        if instr.format.name == 'I':
            imm = random.randint(-2048, 2047) & 0xFFF
            enc = EncodedInstruction(instr, rd=rd, rs1=rs1, imm=imm)
        else:
            enc = EncodedInstruction(instr, rd=rd, rs1=rs1, rs2=rs2)

        self.reg_fsm.mark_written(rd, pc)
        return enc

    def _generate_muldiv_instruction(self, pc: int) -> EncodedInstruction:
        """Generate a multiply/divide instruction."""
        muldiv_instrs = self.instruction_pools.get('muldiv', [])
        if not muldiv_instrs:
            return self._generate_alu_instruction(pc)

        instr = random.choice(muldiv_instrs)
        rd = self._select_dest_register()
        rs1 = self.reg_fsm.select_operand_register(self.config.recent_register_bias)
        rs2 = self.reg_fsm.select_operand_register(self.config.recent_register_bias)

        enc = EncodedInstruction(instr, rd=rd, rs1=rs1, rs2=rs2)
        self.reg_fsm.mark_written(rd, pc)
        return enc

    def _generate_regfsm_instruction(self, pc: int) -> EncodedInstruction:
        """Generate a register setup instruction (LUI or AUIPC)."""
        rd = self._select_dest_register()
        imm = random.randint(0, 0xFFFFF) << 12  # Upper 20 bits

        enc = EncodedInstruction(LUI, rd=rd, imm=imm)
        self.reg_fsm.transition_lui(rd, imm >> 12, pc)
        return enc

    def _generate_mem_instruction(self, pc: int) -> Tuple[EncodedInstruction,
                                                          Optional[CFAmbiguousMarker]]:
        """Generate a memory instruction."""
        mem_instrs = self.instruction_pools.get('mem', [])
        if not mem_instrs:
            return self._generate_alu_instruction(pc), None

        instr = random.choice(mem_instrs)

        if instr.is_load:
            return self._generate_load_instruction(pc, instr)
        else:
            return self._generate_store_instruction(pc, instr), None

    def _generate_load_instruction(self, pc: int,
                                   instr: Instruction) -> Tuple[EncodedInstruction,
                                                                Optional[CFAmbiguousMarker]]:
        """Generate a load instruction."""
        rd = self._select_dest_register()

        base_reg = self._select_operand_register()
        base_value = self.reg_fsm.get_info(base_reg).applied_value
        if base_value is None or not self._addr_in_data(base_value):
            base_reg = self.data_base_reg
            base_value = self.reg_fsm.get_info(base_reg).applied_value or self.memory.layout.data_start

        offset = self._pick_data_offset(base_value, instr.mem_size)

        enc = EncodedInstruction(instr, rd=rd, rs1=base_reg, imm=offset & 0xFFF)
        self.reg_fsm.mark_written(rd, pc)

        # Load is cf-ambiguous (value affects subsequent control flow)
        marker = CFAmbiguousMarker(
            pc=pc,
            instruction=instr,
            encoded=enc,
            target_reg=rd,
            target_value=0,  # Will be determined by ISS
            is_taken=False
        )

        return enc, marker

    def _generate_store_instruction(self, pc: int,
                                    instr: Instruction) -> EncodedInstruction:
        """Generate a store instruction."""
        rs2 = self._select_operand_register()

        base_reg = self._select_operand_register()
        base_value = self.reg_fsm.get_info(base_reg).applied_value
        if base_value is None or not self._addr_in_data(base_value):
            base_reg = self.data_base_reg
            base_value = self.reg_fsm.get_info(base_reg).applied_value or self.memory.layout.data_start

        offset = self._pick_data_offset(base_value, instr.mem_size)

        enc = EncodedInstruction(instr, rs1=base_reg, rs2=rs2, imm=offset & 0xFFF)
        return enc

    def _generate_jal_instruction(self, pc: int) -> Tuple[EncodedInstruction,
                                                          Optional[CFAmbiguousMarker]]:
        """Generate a JAL instruction."""
        rd = self._select_dest_register()

        # Offset will be fixed up later when we know target addresses
        # For now, use a placeholder offset
        offset = 4  # Jump to next instruction as placeholder

        enc = EncodedInstruction(JAL, rd=rd, imm=offset)
        self.reg_fsm.mark_written(rd, pc, value=pc + 4)  # Link address

        return enc, None

    def _generate_jalr_instruction(self, pc: int) -> Tuple[EncodedInstruction,
                                                           Optional[CFAmbiguousMarker]]:
        """Generate a JALR instruction."""
        rd = self._select_dest_register()
        rs1 = self._select_operand_register(allow_zero=False)

        # JALR is cf-ambiguous: target depends on rs1 value
        offset = 0

        enc = EncodedInstruction(JALR, rd=rd, rs1=rs1, imm=offset)
        self.reg_fsm.mark_written(rd, pc, value=pc + 4)

        marker = CFAmbiguousMarker(
            pc=pc,
            instruction=JALR,
            encoded=enc,
            target_reg=rs1,
            target_value=0,  # Will be set to proper jump target
            is_taken=True
        )

        return enc, marker

    def _generate_branch_instruction(self, pc: int) -> Tuple[EncodedInstruction,
                                                              Optional[CFAmbiguousMarker]]:
        """Generate a branch instruction."""
        branch_instrs = [BEQ, BNE, BLT, BGE, BLTU, BGEU]
        instr = random.choice(branch_instrs)

        rs1 = self._select_operand_register(allow_zero=False)
        rs2 = self._select_operand_register(allow_zero=False)

        # Decide if branch should be taken or not
        is_taken = random.choice([True, False])

        # Offset will be fixed up later
        offset = 4 if not is_taken else 8  # Placeholder

        enc = EncodedInstruction(instr, rs1=rs1, rs2=rs2, imm=offset)

        marker = CFAmbiguousMarker(
            pc=pc,
            instruction=instr,
            encoded=enc,
            target_reg=rs1,  # We'll need to set up rs1 and rs2
            target_value=0,
            is_taken=is_taken
        )

        return enc, marker

    def _select_dest_register(self) -> int:
        """Select a destination register."""
        # Avoid x0
        while True:
            reg = random.randint(1, self.config.cpu.num_gpr - 1)
            if reg not in self.reserved_regs:
                return reg

    def _select_operand_register(self, allow_zero: bool = True) -> int:
        """Select an operand register avoiding reserved registers."""
        for _ in range(10):
            reg = self.reg_fsm.select_operand_register(self.config.recent_register_bias)
            if reg in self.reserved_regs:
                continue
            if not allow_zero and reg == 0:
                continue
            return reg
        for reg in range(1, self.config.cpu.num_gpr):
            if reg not in self.reserved_regs:
                return reg
        return 1

    def generate_initial_block(self, start_addr: int) -> BasicBlock:
        """
        Generate the initial basic block.

        Sets up initial state:
        - Stack pointer
        - Random register values
        """
        block = BasicBlock(start_addr=start_addr, block_id=0)
        pc = start_addr

        # Set up stack pointer (x2)
        sp_value = self.memory.get_stack_pointer()
        sp_upper = (sp_value + 0x800) >> 12
        sp_lower = sp_value - (sp_upper << 12)

        block.instructions.append(EncodedInstruction(LUI, rd=2, imm=sp_upper << 12))
        self.reg_fsm.transition_lui(2, sp_upper, pc)
        pc += 4

        block.instructions.append(EncodedInstruction(ADDI, rd=2, rs1=2, imm=sp_lower & 0xFFF))
        self.reg_fsm.transition_addi_complete(2, sp_lower & 0xFFF, pc)
        pc += 4

        # Initialize some registers with random values
        for reg in range(3, 16):
            value = random.randint(0, 0xFFFFFFFF)
            upper = (value + 0x800) >> 12
            lower = value - (upper << 12)

            block.instructions.append(EncodedInstruction(LUI, rd=reg, imm=(upper & 0xFFFFF) << 12))
            self.reg_fsm.transition_lui(reg, upper & 0xFFFFF, pc)
            pc += 4

            block.instructions.append(EncodedInstruction(ADDI, rd=reg, rs1=reg, imm=lower & 0xFFF))
            self.reg_fsm.transition_addi_complete(reg, lower & 0xFFF, pc)
            pc += 4

        # Initialize data base register
        data_base = self.memory.layout.data_start
        upper = (data_base + 0x800) >> 12
        lower = data_base - (upper << 12)
        block.instructions.append(EncodedInstruction(LUI, rd=self.data_base_reg, imm=(upper & 0xFFFFF) << 12))
        self.reg_fsm.transition_lui(self.data_base_reg, upper & 0xFFFFF, pc)
        pc += 4
        block.instructions.append(EncodedInstruction(ADDI, rd=self.data_base_reg, rs1=self.data_base_reg, imm=lower & 0xFFF))
        self.reg_fsm.transition_addi_complete(self.data_base_reg, lower & 0xFFF, pc)
        pc += 4

        # Jump to next block (placeholder)
        block.terminator = EncodedInstruction(JAL, rd=0, imm=4)

        return block

    def generate_final_block(self, start_addr: int, block_id: int) -> BasicBlock:
        """
        Generate the final basic block.

        Signals completion with a trap (ebreak).
        """
        block = BasicBlock(start_addr=start_addr, block_id=block_id)

        if self.config.cpu.name == "kronos":
            # Write to tohost (data_start) for kronos_compliance completion.
            block.instructions.append(EncodedInstruction(ADDI, rd=1, rs1=0, imm=1))
            block.instructions.append(EncodedInstruction(SW, rs1=self.data_base_reg, rs2=1, imm=0))
            block.terminator = EncodedInstruction(EBREAK)
        else:
            # Terminate by triggering a trap so ISS/RTL can stop.
            block.terminator = EncodedInstruction(EBREAK)

        return block

    def _addr_in_data(self, addr: int) -> bool:
        """Check if address falls within data region."""
        start = self.memory.layout.data_start
        end = self.memory.layout.data_start + self.memory.layout.data_size
        return start <= addr < end

    def _pick_data_offset(self, base_value: int, size: int) -> int:
        """Pick a small offset so base+offset stays in data region."""
        start = self.memory.layout.data_start
        end = self.memory.layout.data_start + self.memory.layout.data_size
        min_offset = start - base_value
        max_offset = (end - size) - base_value
        if min_offset > max_offset:
            return 0
        min_offset = max(min_offset, -2048)
        max_offset = min(max_offset, 2047)
        if min_offset > max_offset:
            return 0
        alignment = max(1, int(size))
        base_mod = (base_value + min_offset) % alignment
        if base_mod != 0:
            min_offset += alignment - base_mod
        max_offset -= (base_value + max_offset) % alignment
        if min_offset > max_offset:
            return 0
        return random.randrange(int(min_offset), int(max_offset) + 1, alignment)
