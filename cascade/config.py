"""
Configuration and CPU parameters for Cascade fuzzer.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set
from pathlib import Path


class Extension(Enum):
    """RISC-V ISA extensions."""
    I = auto()   # Base integer
    M = auto()   # Multiplication/Division
    A = auto()   # Atomics
    F = auto()   # Single-precision floating-point
    D = auto()   # Double-precision floating-point
    C = auto()   # Compressed


class PrivilegeLevel(Enum):
    """RISC-V privilege levels."""
    USER = 0
    SUPERVISOR = 1
    MACHINE = 3


@dataclass
class MemoryRegion:
    """Represents a memory region."""
    start: int
    size: int

    @property
    def end(self) -> int:
        return self.start + self.size

    def contains(self, addr: int) -> bool:
        return self.start <= addr < self.end

    def overlaps(self, other: 'MemoryRegion') -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class MemoryLayout:
    """Memory layout configuration."""
    # Code section
    code_start: int = 0x80000000
    code_size: int = 0x10000  # 64KB

    # Data section (for stores)
    data_start: int = 0x80010000
    data_size: int = 0x10000  # 64KB

    # Stack
    stack_top: int = 0x80030000
    stack_size: int = 0x10000  # 64KB

    # Context setter block (for reduction)
    context_start: int = 0x80040000
    context_size: int = 0x1000  # 4KB


KRONOS_MEMORY_LAYOUT = MemoryLayout(
    code_start=0x00000000,
    code_size=0x1000,
    data_start=0x00001000,
    data_size=0x1000,
    stack_top=0x00002000,
    stack_size=0x800,
    context_start=0x00001800,
    context_size=0x800,
)


@dataclass
class InstructionWeights:
    """Probability weights for instruction categories."""
    # Phase 1: rv32i base
    alu: float = 0.25
    mem: float = 0.15
    branch: float = 0.15
    jal: float = 0.05
    jalr: float = 0.05
    regfsm: float = 0.15  # Register setup (lui, addi)
    csr: float = 0.05
    fence: float = 0.02

    # Phase 2: M extension
    muldiv: float = 0.08

    # Phase 3: Privileged
    exception: float = 0.02
    privilege: float = 0.03

    def normalize(self) -> Dict[str, float]:
        """Return normalized weights summing to 1.0."""
        total = (self.alu + self.mem + self.branch + self.jal +
                 self.jalr + self.regfsm + self.csr + self.fence +
                 self.muldiv + self.exception + self.privilege)
        if total == 0:
            return {}
        return {
            'alu': self.alu / total,
            'mem': self.mem / total,
            'branch': self.branch / total,
            'jal': self.jal / total,
            'jalr': self.jalr / total,
            'regfsm': self.regfsm / total,
            'csr': self.csr / total,
            'fence': self.fence / total,
            'muldiv': self.muldiv / total,
            'exception': self.exception / total,
            'privilege': self.privilege / total,
        }


@dataclass
class CPUConfig:
    """CPU-specific configuration."""
    name: str
    xlen: int = 32  # 32 or 64
    extensions: Set[Extension] = field(default_factory=lambda: {Extension.I})

    # Available privilege levels
    privilege_levels: Set[PrivilegeLevel] = field(
        default_factory=lambda: {PrivilegeLevel.MACHINE}
    )

    # Number of general-purpose registers (always 32 for RISC-V)
    num_gpr: int = 32

    # Physical memory protection regions
    pmp_regions: int = 0

    # Hardware performance counters
    hpm_counters: int = 0

    # CSR availability
    available_csrs: Set[int] = field(default_factory=set)

    # Known bugs to circumvent
    bug_circumventions: List[str] = field(default_factory=list)


@dataclass
class FuzzerConfig:
    """Main fuzzer configuration."""
    # Target CPU
    cpu: CPUConfig = field(default_factory=lambda: CPUConfig(name="default"))

    # Memory layout
    memory: MemoryLayout = field(default_factory=MemoryLayout)

    # Instruction weights
    weights: InstructionWeights = field(default_factory=InstructionWeights)

    # Program generation parameters
    min_basic_blocks: int = 10
    max_basic_blocks: int = 100
    min_block_instructions: int = 1
    max_block_instructions: int = 20

    # Register bias: probability of selecting recently-written register
    recent_register_bias: float = 0.7

    # Execution parameters
    iss_timeout: int = 20000  # ISS timeout in ms
    rtl_timeout: int = 20000  # RTL timeout in ms

    # Tool paths
    spike_path: Path = field(default_factory=lambda: Path("/opt/riscv/bin/spike"))
    verilator_path: Path = field(default_factory=lambda: Path("verilator"))

    # RTL model path
    rtl_model_path: Optional[Path] = None

    # Output directory
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    # Random seed (None = random)
    seed: Optional[int] = None

    # Number of programs to generate
    num_programs: int = 1000

    # Parallel workers
    num_workers: int = 1


# Pre-defined CPU configurations
PICORV32_CONFIG = CPUConfig(
    name="picorv32",
    xlen=32,
    extensions={Extension.I, Extension.M},
    privilege_levels={PrivilegeLevel.MACHINE},
)

KRONOS_CONFIG = CPUConfig(
    name="kronos",
    xlen=32,
    extensions={Extension.I},
    privilege_levels={PrivilegeLevel.MACHINE},
)

VEXRISCV_CONFIG = CPUConfig(
    name="vexriscv",
    xlen=32,
    extensions={Extension.I, Extension.M, Extension.F, Extension.D},
    privilege_levels={PrivilegeLevel.MACHINE, PrivilegeLevel.USER},
)

CVA6_CONFIG = CPUConfig(
    name="cva6",
    xlen=64,
    extensions={Extension.I, Extension.M, Extension.A, Extension.F, Extension.D},
    privilege_levels={PrivilegeLevel.MACHINE, PrivilegeLevel.SUPERVISOR, PrivilegeLevel.USER},
)

ROCKET_CONFIG = CPUConfig(
    name="rocket",
    xlen=64,
    extensions={Extension.I, Extension.M, Extension.A, Extension.F, Extension.D},
    privilege_levels={PrivilegeLevel.MACHINE, PrivilegeLevel.SUPERVISOR, PrivilegeLevel.USER},
    bug_circumventions=["no_instret_after_ecall"],  # Known bug R1 from paper
)

BOOM_CONFIG = CPUConfig(
    name="boom",
    xlen=64,
    extensions={Extension.I, Extension.M, Extension.A, Extension.F, Extension.D},
    privilege_levels={PrivilegeLevel.MACHINE, PrivilegeLevel.SUPERVISOR, PrivilegeLevel.USER},
)


def load_config(config_path: Optional[Path] = None) -> FuzzerConfig:
    """Load configuration from file or return default."""
    if config_path is None:
        return FuzzerConfig()

    # TODO: Implement config file loading (YAML/JSON)
    raise NotImplementedError("Config file loading not yet implemented")
