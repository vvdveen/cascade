# macOS Port Instructions for Cascade

## Instructions for LLM

You are tasked with adapting Cascade, a RISC-V CPU fuzzer, to run on macOS. The project is already implemented and working on Linux.

**Repository**: https://github.com/vvdveen/cascade

Clone it and work from there.

---

## Background

### What is Cascade?

Cascade is a RISC-V CPU fuzzer that generates valid programs with entangled data and control flows. It detects bugs via program non-termination (timeout = bug). The key innovation is **asymmetric ISA pre-simulation** - using an ISS (Instruction Set Simulator) to construct valid programs rather than for differential comparison.

### Architecture

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
└─────────────────────────────────────────────────────────────────────┘
```

### Key Dependencies

1. **Python 3.10+** - Core fuzzer implementation
2. **Spike** (riscv-isa-sim) - RISC-V Instruction Set Simulator for pre-simulation
3. **Verilator 5.x** - RTL simulation (converts Verilog to C++)
4. **PicoRV32** - Simple RISC-V CPU for testing (optional)
5. **Kronos** - rv32i CPU for FPGA-focused testing (optional)

---

## What Needs to Change for macOS

### 1. Setup Script (`scripts/setup_deps.sh`)

The current script uses `apt-get` (Debian/Ubuntu). For macOS, you need to:

- Use **Homebrew** (`brew`) instead of apt-get
- Different package names:
  - `build-essential` → Xcode Command Line Tools (`xcode-select --install`)
  - `device-tree-compiler` → `dtc`
  - `libboost-all-dev` → `boost`
  - etc.
- Handle Apple Silicon (arm64) vs Intel (x86_64) differences
- Spike and Verilator may need different build flags on macOS

### 2. macOS-Specific Packages via Homebrew

```bash
# Likely needed packages
brew install autoconf automake libtool pkg-config
brew install boost dtc
brew install verilator  # Homebrew has verilator 5.x
```

### 3. Spike Build on macOS

Spike may need adjustments:
- Use `glibtoolize` instead of `libtoolize` (Homebrew installs GNU libtool as glibtool)
- May need to set `CXXFLAGS` for Apple Clang compatibility
- Boost paths may differ: `--with-boost=$(brew --prefix boost)`

### 4. Verilator on macOS

Homebrew's Verilator should work, but if building from source:
- Needs `bison` from Homebrew (macOS bison is too old): `brew install bison` and add to PATH
- May need: `export PATH="$(brew --prefix bison)/bin:$PATH"`

### 5. Python Virtual Environment

Same as Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Files to Modify

1. **`scripts/setup_deps.sh`** - Add macOS support with Homebrew
   - Detect OS (`uname -s`)
   - Use `brew` on Darwin (macOS)
   - Handle different package names
   - Handle Apple Silicon vs Intel if needed

2. **`README.md`** - Add macOS installation instructions

3. **Test on macOS** - Run `pytest tests/` to verify

---

## Current Project Structure

```
cascade/
├── cascade/
│   ├── __init__.py
│   ├── config.py              # Configuration & CPU parameters
│   ├── fuzzer.py              # Main fuzzing loop
│   ├── isa/                   # RISC-V instruction definitions
│   ├── generator/             # Program generation
│   ├── execution/             # ISS and RTL runners
│   └── reduction/             # Bug-triggering program reduction
├── tests/                     # 62 passing tests
├── scripts/
│   └── setup_deps.sh          # THIS NEEDS macOS SUPPORT
├── pyproject.toml
└── README.md
```

---

## Testing Your Changes

```bash
# 1. Clone the repo
git clone https://github.com/vvdveen/cascade.git
cd cascade

# 2. Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Run tests (RTL integration tests require a built simulator)
make -C deps/picorv32 sim
export CASCADE_RTL_PATH=deps/picorv32
pytest tests/

# Optional: Run RTL integration tests against Kronos
./scripts/setup_deps.sh kronos
CASCADE_RTL_PATH=deps/kronos CASCADE_RTL_CPU=kronos pytest tests/test_rtl_integration.py

# 4. Test the fuzzer with mock runners
cascade -n 10 --cpu picorv32
cascade -n 10 --cpu kronos

# 5. If you install Spike, test with real ISS
cascade -n 10 --cpu picorv32 --spike-path /opt/homebrew/bin/spike
cascade -n 10 --cpu kronos --spike-path /opt/homebrew/bin/spike
```

---

## Disassembling Bug Artifacts (objdump)

To disassemble `ultimate.elf` for a bug:

```bash
riscv32-unknown-elf-objdump -d output/bugs/<bug_id>/ultimate.elf
```

On macOS, install the RISC-V toolchain via Homebrew (tap required):

```bash
brew tap riscv/riscv
brew install riscv/riscv/riscv-gnu-toolchain
```

If your Homebrew package only provides `riscv64-unknown-elf-objdump`, use that instead.

---

## Expected Deliverables

1. Modified `scripts/setup_deps.sh` that works on macOS (Homebrew)
2. Updated `README.md` with macOS instructions
3. All tests passing on macOS (RTL integration tests require a built simulator)
4. Fuzzer running (at minimum with mock runners)

---

## Notes

- The fuzzer works with **mock runners** even without Spike/Verilator installed - useful for testing
- Focus on getting the Python code and basic tooling working first
- Spike and Verilator are optional but needed for real CPU fuzzing
- On Apple Silicon, Spike should work but some RTL models might have issues (cross that bridge later)

Good luck!
