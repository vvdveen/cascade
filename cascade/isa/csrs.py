"""
RISC-V Control and Status Register (CSR) definitions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set, Optional


class CSRAccess(Enum):
    """CSR access permissions."""
    RW = "rw"   # Read-write
    RO = "ro"   # Read-only


class CSRPrivilege(Enum):
    """CSR privilege level required for access."""
    USER = 0
    SUPERVISOR = 1
    HYPERVISOR = 2
    MACHINE = 3


@dataclass
class CSR:
    """CSR definition."""
    name: str
    address: int
    access: CSRAccess
    privilege: CSRPrivilege
    description: str = ""

    # Bit mask for writable bits (0xFFFFFFFF = all bits writable)
    writable_mask: int = 0xFFFFFFFF

    # Initial/reset value
    reset_value: int = 0

    def is_readable(self) -> bool:
        return True  # All CSRs are readable

    def is_writable(self) -> bool:
        return self.access == CSRAccess.RW


# Machine-level CSRs

# Machine Information Registers
MVENDORID = CSR("mvendorid", 0xF11, CSRAccess.RO, CSRPrivilege.MACHINE,
                "Vendor ID")
MARCHID = CSR("marchid", 0xF12, CSRAccess.RO, CSRPrivilege.MACHINE,
              "Architecture ID")
MIMPID = CSR("mimpid", 0xF13, CSRAccess.RO, CSRPrivilege.MACHINE,
             "Implementation ID")
MHARTID = CSR("mhartid", 0xF14, CSRAccess.RO, CSRPrivilege.MACHINE,
              "Hardware thread ID")
MCONFIGPTR = CSR("mconfigptr", 0xF15, CSRAccess.RO, CSRPrivilege.MACHINE,
                 "Configuration pointer")

# Machine Trap Setup
MSTATUS = CSR("mstatus", 0x300, CSRAccess.RW, CSRPrivilege.MACHINE,
              "Machine status", writable_mask=0x807FF9BB)
MISA = CSR("misa", 0x301, CSRAccess.RW, CSRPrivilege.MACHINE,
           "ISA and extensions")
MEDELEG = CSR("medeleg", 0x302, CSRAccess.RW, CSRPrivilege.MACHINE,
              "Machine exception delegation")
MIDELEG = CSR("mideleg", 0x303, CSRAccess.RW, CSRPrivilege.MACHINE,
              "Machine interrupt delegation")
MIE = CSR("mie", 0x304, CSRAccess.RW, CSRPrivilege.MACHINE,
          "Machine interrupt enable")
MTVEC = CSR("mtvec", 0x305, CSRAccess.RW, CSRPrivilege.MACHINE,
            "Machine trap-handler base address")
MCOUNTEREN = CSR("mcounteren", 0x306, CSRAccess.RW, CSRPrivilege.MACHINE,
                 "Machine counter enable")
MSTATUSH = CSR("mstatush", 0x310, CSRAccess.RW, CSRPrivilege.MACHINE,
               "Machine status (high bits, RV32 only)")

# Machine Trap Handling
MSCRATCH = CSR("mscratch", 0x340, CSRAccess.RW, CSRPrivilege.MACHINE,
               "Machine scratch register")
MEPC = CSR("mepc", 0x341, CSRAccess.RW, CSRPrivilege.MACHINE,
           "Machine exception PC")
MCAUSE = CSR("mcause", 0x342, CSRAccess.RW, CSRPrivilege.MACHINE,
             "Machine trap cause")
MTVAL = CSR("mtval", 0x343, CSRAccess.RW, CSRPrivilege.MACHINE,
            "Machine trap value")
MIP = CSR("mip", 0x344, CSRAccess.RW, CSRPrivilege.MACHINE,
          "Machine interrupt pending")

# Machine Memory Protection
PMPCFG0 = CSR("pmpcfg0", 0x3A0, CSRAccess.RW, CSRPrivilege.MACHINE,
              "PMP config 0")
PMPCFG1 = CSR("pmpcfg1", 0x3A1, CSRAccess.RW, CSRPrivilege.MACHINE,
              "PMP config 1 (RV32 only)")
PMPCFG2 = CSR("pmpcfg2", 0x3A2, CSRAccess.RW, CSRPrivilege.MACHINE,
              "PMP config 2")
PMPCFG3 = CSR("pmpcfg3", 0x3A3, CSRAccess.RW, CSRPrivilege.MACHINE,
              "PMP config 3 (RV32 only)")

PMPADDR0 = CSR("pmpaddr0", 0x3B0, CSRAccess.RW, CSRPrivilege.MACHINE,
               "PMP address 0")
PMPADDR1 = CSR("pmpaddr1", 0x3B1, CSRAccess.RW, CSRPrivilege.MACHINE,
               "PMP address 1")
PMPADDR2 = CSR("pmpaddr2", 0x3B2, CSRAccess.RW, CSRPrivilege.MACHINE,
               "PMP address 2")
PMPADDR3 = CSR("pmpaddr3", 0x3B3, CSRAccess.RW, CSRPrivilege.MACHINE,
               "PMP address 3")

# Machine Counters/Timers
MCYCLE = CSR("mcycle", 0xB00, CSRAccess.RW, CSRPrivilege.MACHINE,
             "Machine cycle counter")
MINSTRET = CSR("minstret", 0xB02, CSRAccess.RW, CSRPrivilege.MACHINE,
               "Machine instructions retired")
MCYCLEH = CSR("mcycleh", 0xB80, CSRAccess.RW, CSRPrivilege.MACHINE,
              "Machine cycle counter (high bits, RV32 only)")
MINSTRETH = CSR("minstreth", 0xB82, CSRAccess.RW, CSRPrivilege.MACHINE,
                "Machine instructions retired (high bits, RV32 only)")

# User-level CSRs (read-only shadows)
CYCLE = CSR("cycle", 0xC00, CSRAccess.RO, CSRPrivilege.USER,
            "Cycle counter")
TIME = CSR("time", 0xC01, CSRAccess.RO, CSRPrivilege.USER,
           "Timer")
INSTRET = CSR("instret", 0xC02, CSRAccess.RO, CSRPrivilege.USER,
              "Instructions retired")
CYCLEH = CSR("cycleh", 0xC80, CSRAccess.RO, CSRPrivilege.USER,
             "Cycle counter (high bits, RV32 only)")
TIMEH = CSR("timeh", 0xC81, CSRAccess.RO, CSRPrivilege.USER,
            "Timer (high bits, RV32 only)")
INSTRETH = CSR("instreth", 0xC82, CSRAccess.RO, CSRPrivilege.USER,
               "Instructions retired (high bits, RV32 only)")

# Supervisor-level CSRs
SSTATUS = CSR("sstatus", 0x100, CSRAccess.RW, CSRPrivilege.SUPERVISOR,
              "Supervisor status")
SIE = CSR("sie", 0x104, CSRAccess.RW, CSRPrivilege.SUPERVISOR,
          "Supervisor interrupt enable")
STVEC = CSR("stvec", 0x105, CSRAccess.RW, CSRPrivilege.SUPERVISOR,
            "Supervisor trap handler base address")
SCOUNTEREN = CSR("scounteren", 0x106, CSRAccess.RW, CSRPrivilege.SUPERVISOR,
                 "Supervisor counter enable")
SSCRATCH = CSR("sscratch", 0x140, CSRAccess.RW, CSRPrivilege.SUPERVISOR,
               "Supervisor scratch register")
SEPC = CSR("sepc", 0x141, CSRAccess.RW, CSRPrivilege.SUPERVISOR,
           "Supervisor exception PC")
SCAUSE = CSR("scause", 0x142, CSRAccess.RW, CSRPrivilege.SUPERVISOR,
             "Supervisor trap cause")
STVAL = CSR("stval", 0x143, CSRAccess.RW, CSRPrivilege.SUPERVISOR,
            "Supervisor trap value")
SIP = CSR("sip", 0x144, CSRAccess.RW, CSRPrivilege.SUPERVISOR,
          "Supervisor interrupt pending")
SATP = CSR("satp", 0x180, CSRAccess.RW, CSRPrivilege.SUPERVISOR,
           "Supervisor address translation and protection")


# All CSR definitions
CSR_DEFINITIONS: Dict[int, CSR] = {
    # Machine Information
    0xF11: MVENDORID,
    0xF12: MARCHID,
    0xF13: MIMPID,
    0xF14: MHARTID,
    0xF15: MCONFIGPTR,

    # Machine Trap Setup
    0x300: MSTATUS,
    0x301: MISA,
    0x302: MEDELEG,
    0x303: MIDELEG,
    0x304: MIE,
    0x305: MTVEC,
    0x306: MCOUNTEREN,
    0x310: MSTATUSH,

    # Machine Trap Handling
    0x340: MSCRATCH,
    0x341: MEPC,
    0x342: MCAUSE,
    0x343: MTVAL,
    0x344: MIP,

    # Machine Memory Protection
    0x3A0: PMPCFG0,
    0x3A1: PMPCFG1,
    0x3A2: PMPCFG2,
    0x3A3: PMPCFG3,
    0x3B0: PMPADDR0,
    0x3B1: PMPADDR1,
    0x3B2: PMPADDR2,
    0x3B3: PMPADDR3,

    # Machine Counters
    0xB00: MCYCLE,
    0xB02: MINSTRET,
    0xB80: MCYCLEH,
    0xB82: MINSTRETH,

    # User Counters
    0xC00: CYCLE,
    0xC01: TIME,
    0xC02: INSTRET,
    0xC80: CYCLEH,
    0xC81: TIMEH,
    0xC82: INSTRETH,

    # Supervisor
    0x100: SSTATUS,
    0x104: SIE,
    0x105: STVEC,
    0x106: SCOUNTEREN,
    0x140: SSCRATCH,
    0x141: SEPC,
    0x142: SCAUSE,
    0x143: STVAL,
    0x144: SIP,
    0x180: SATP,
}


def get_csr(address: int) -> Optional[CSR]:
    """Get CSR definition by address."""
    return CSR_DEFINITIONS.get(address)


def get_csrs_for_privilege(privilege: CSRPrivilege) -> Dict[int, CSR]:
    """Get all CSRs accessible at given privilege level."""
    return {
        addr: csr for addr, csr in CSR_DEFINITIONS.items()
        if csr.privilege.value <= privilege.value
    }


def get_writable_csrs(privilege: CSRPrivilege) -> Dict[int, CSR]:
    """Get all writable CSRs at given privilege level."""
    return {
        addr: csr for addr, csr in CSR_DEFINITIONS.items()
        if csr.is_writable() and csr.privilege.value <= privilege.value
    }


# Machine Status Register (mstatus) bit definitions
class MStatusBits:
    """mstatus register bit positions."""
    UIE = 0      # User interrupt enable
    SIE = 1      # Supervisor interrupt enable
    MIE = 3      # Machine interrupt enable
    UPIE = 4     # User previous interrupt enable
    SPIE = 5     # Supervisor previous interrupt enable
    MPIE = 7     # Machine previous interrupt enable
    SPP = 8      # Supervisor previous privilege
    MPP = 11     # Machine previous privilege (2 bits: 11-12)
    FS = 13      # Floating-point status (2 bits: 13-14)
    XS = 15      # User extension status (2 bits: 15-16)
    MPRV = 17    # Modify privilege
    SUM = 18     # Supervisor user memory access
    MXR = 19     # Make executable readable
    TVM = 20     # Trap virtual memory
    TW = 21      # Timeout wait
    TSR = 22     # Trap SRET
    SD = 31      # Dirty (RV32) / 63 (RV64)


# Exception cause codes
class ExceptionCause:
    """Exception cause codes."""
    INSTRUCTION_MISALIGNED = 0
    INSTRUCTION_ACCESS_FAULT = 1
    ILLEGAL_INSTRUCTION = 2
    BREAKPOINT = 3
    LOAD_MISALIGNED = 4
    LOAD_ACCESS_FAULT = 5
    STORE_MISALIGNED = 6
    STORE_ACCESS_FAULT = 7
    ECALL_FROM_U = 8
    ECALL_FROM_S = 9
    ECALL_FROM_M = 11
    INSTRUCTION_PAGE_FAULT = 12
    LOAD_PAGE_FAULT = 13
    STORE_PAGE_FAULT = 15


# Interrupt cause codes (bit 31 set)
class InterruptCause:
    """Interrupt cause codes."""
    USER_SOFTWARE = 0
    SUPERVISOR_SOFTWARE = 1
    MACHINE_SOFTWARE = 3
    USER_TIMER = 4
    SUPERVISOR_TIMER = 5
    MACHINE_TIMER = 7
    USER_EXTERNAL = 8
    SUPERVISOR_EXTERNAL = 9
    MACHINE_EXTERNAL = 11
