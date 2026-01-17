# Cascade

A RISC-V CPU fuzzer with asymmetric ISA pre-simulation.

## Overview

Cascade generates valid RISC-V programs with entangled data and control flows. Bugs are detected via program non-termination (timeout = bug), eliminating runtime overhead for differential comparison.

Key innovation: **Asymmetric ISA Pre-simulation** - Instead of using an ISS (Instruction Set Simulator) for differential comparison, Cascade uses it to *construct* valid programs by collecting register values at control-flow decision points.

## Features

- Generates complex RISC-V programs with interleaved data and control dependencies
- XOR-based offset construction for entangling data flow with control flow
- Timeout-based bug detection (no runtime overhead)
- Automatic program reduction for bug-triggering programs
- Support for multiple RISC-V CPUs (PicoRV32, VexRiscv, Rocket, CVA6, BOOM)

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-repo/cascade.git
cd cascade

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Cascade and dependencies
pip install -e ".[dev]"

# (Optional) Install external tools (Spike, Verilator, PicoRV32)
./scripts/setup_deps.sh
```

### macOS (Homebrew)

```bash
# Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# Install dependencies
brew install autoconf automake libtool pkg-config boost dtc bison flex ccache gperftools

# Optional external tools (Spike from source, Verilator via Homebrew)
# If setup_deps.sh reports missing autoconf/automake, install:
# brew install autoconf automake libtool pkg-config
./scripts/setup_deps.sh
```

### Manual Installation

1. **Install Spike (RISC-V ISS)**:
```bash
git clone https://github.com/riscv-software-src/riscv-isa-sim.git
cd riscv-isa-sim
mkdir build && cd build
../configure --prefix=/opt/riscv
make -j$(nproc)  # macOS: use `sysctl -n hw.ncpu`
sudo make install
```

2. **Install Verilator (5.x+)**:
```bash
git clone https://github.com/verilator/verilator.git
cd verilator
git checkout v5.005
autoconf && ./configure
make -j$(nproc)
sudo make install
```

3. **Install Cascade**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

### Basic Fuzzing

```bash
# Run with default settings (PicoRV32, 100 programs)
cascade -n 100 --cpu picorv32

# Specify RTL model path
cascade -n 1000 --cpu picorv32 --rtl-path /path/to/picorv32

# Use specific seed for reproducibility
cascade -n 100 --seed 12345

# Verbose output
cascade -n 100 -v
```

### Python API

```python
from cascade.config import FuzzerConfig, PICORV32_CONFIG
from cascade.fuzzer import Fuzzer

# Create configuration
config = FuzzerConfig(
    cpu=PICORV32_CONFIG,
    num_programs=100,
)

# Create and run fuzzer
fuzzer = Fuzzer(config)
fuzzer.calibrate()
bugs = fuzzer.fuzz()

for bug in bugs:
    print(f"Bug found: {bug.bug_id}")
    print(f"  Program: {bug.program_path}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Cascade Fuzzer                                │
├─────────────────────────────────────────────────────────────────────┤
│  1. Intermediate Program    2. Ultimate Program    3. Execution      │
│     Construction               Construction           & Detection    │
│  ┌──────────────────┐      ┌──────────────────┐   ┌──────────────┐  │
│  │ Basic Block Gen  │ ───► │ ISS Pre-sim      │ ─►│ RTL Sim      │  │
│  │ Memory Manager   │      │ Feedback Integr. │   │ (Verilator)  │  │
│  │ Register FSMs    │      │ Offset Construct │   │ Timeout=Bug  │  │
│  └──────────────────┘      └──────────────────┘   └──────────────┘  │
│                                                                      │
│  4. Program Reduction (for bug-triggering programs)                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Tail Detection → Head Detection → Minimal Program            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
cascade/
├── cascade/
│   ├── config.py           # Configuration & CPU parameters
│   ├── fuzzer.py           # Main fuzzing loop
│   ├── isa/
│   │   ├── instructions.py # RISC-V instruction definitions
│   │   ├── encoding.py     # Instruction encoding/decoding
│   │   ├── csrs.py         # Control & Status Registers
│   │   └── extensions.py   # ISA extension support
│   ├── generator/
│   │   ├── memory_manager.py   # Memory allocation
│   │   ├── register_fsm.py     # Register state machine
│   │   ├── basic_block.py      # Basic block generation
│   │   ├── intermediate.py     # Intermediate program
│   │   └── ultimate.py         # Ultimate program (with offsets)
│   ├── execution/
│   │   ├── iss_runner.py   # Spike ISS interface
│   │   ├── rtl_runner.py   # Verilator interface
│   │   └── elf_writer.py   # ELF file generation
│   └── reduction/
│       ├── tail_finder.py  # Find last bug instruction
│       ├── head_finder.py  # Find first bug instruction
│       └── reducer.py      # Main reduction orchestrator
├── tests/
├── scripts/
│   └── setup_deps.sh       # Dependency installation
├── configs/
└── pyproject.toml
```

## Supported CPUs

| CPU      | ISA Extensions | Notes |
|----------|----------------|-------|
| PicoRV32 | rv32im         | Simple, good for testing |
| VexRiscv | rv32imfd       | SpinalHDL-based |
| Rocket   | rv64imafd      | Reference implementation |
| CVA6     | rv64imafd      | OpenHW Group |
| BOOM     | rv64imafd      | Out-of-order |

## Testing

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=cascade

# Run specific test file
pytest tests/test_encoding.py
```

## How It Works

### 1. Intermediate Program Construction

Cascade generates programs where control flow is isolated from data flow. CF-ambiguous instructions (branches, JALR, loads) use placeholder values.

### 2. ISS Pre-simulation

The intermediate program runs on Spike ISS to collect actual register values at each CF-ambiguous instruction.

### 3. Ultimate Program Construction

Using ISS feedback, Cascade constructs XOR-based offsets:
```
lui  r_off, offset[31:12]
addi r_off, r_off, offset[11:0]
xor  r_app, r_off, r_d    # r_app = target_value
```

The offset is computed as: `offset = target_value XOR r_d_value`

### 4. RTL Execution

The ultimate program runs on the RTL simulator. A timeout indicates a potential bug.

### 5. Program Reduction

Bug-triggering programs are reduced using binary search to find:
- **Tail**: Last instruction involved in the bug
- **Head**: First instruction involved in the bug

## License

MIT License
