# Cascade Debugging Status

## TL;DR

**The fuzzer can run, but ISS timeouts are likely.** Root cause: intermediate programs use JALR and branch instructions but don't set up the register values they read from. This causes control flow to jump to wrong addresses, creating infinite loops. Spike times out because ebreak is never reached.

**Fix options:** In `intermediate.py:_fixup_control_flow()`, either replace JALR/branch terminators with JAL instructions (deterministic), or insert register setup before JALR/branches so the intermediate program is deterministic while preserving CF-ambiguity in the ultimate program.

---

This document brings another LLM or developer up to speed with the current debugging state of the Cascade CPU fuzzer.

## What is Cascade?

Cascade is a RISC-V CPU fuzzer based on the USENIX Security 2024 paper "Cascade: CPU Fuzzing via Intricate Program Generation" by Solt et al. The key innovation is **Asymmetric ISA Pre-simulation**:

1. **Intermediate Program**: Generate a program with CF-ambiguous instructions using placeholder values
2. **ISS Execution**: Run the intermediate program on Spike (RISC-V ISS) to collect register values
3. **Ultimate Program**: Use ISS feedback to construct XOR-based offsets that make control flow deterministic
4. **RTL Execution**: Run the ultimate program on RTL (Verilator). A timeout indicates a bug.

The ISS is NOT used for differential comparison - it's used to *construct* valid programs. Programs terminate with `ebreak` instruction. Bug detection is timeout-based (program should complete, non-termination = bug).

## Current Problem

**Spike (ISS) can time out on generated programs due to non-deterministic control flow in the intermediate program.**

### Symptoms

Running the fuzzer shows:
```
Overall [....................] 0/100000 done 24/100000 bugs 0
```

- `0/100000 done` = no programs executed on RTL
- `24/100000` = 24 programs attempted (completed in fuzzer loop)
- `bugs 0` = no bugs found

Worker logs show:
```
Iteration 0: generated=1, executed=0, iss_errors=1
Iteration 1: generated=2, executed=0, iss_errors=2
```

- `generated` increments (programs are being created)
- `executed` stays at 0 (no programs run on RTL)
- `iss_errors` increments (every ISS execution fails)

The error is: `ISS simulation timed out`

### Why This Is A Problem

Per the paper, ISS timeout is NOT expected behavior. The intermediate program should:
1. Execute deterministically on the ISS
2. Complete with `ebreak` instruction
3. Provide register value feedback for ultimate program construction

Without ISS feedback, the fuzzer cannot reliably:
- Construct XOR-based offsets for CF-ambiguous instructions
- Build ultimate programs
- Run anything on RTL
- Find bugs

## Architecture

```
cascade/
├── cascade/
│   ├── fuzzer.py              # Main fuzzer loop, worker processes
│   ├── config.py              # Configuration, CPU parameters
│   ├── generator/
│   │   ├── intermediate.py    # Intermediate program construction
│   │   ├── ultimate.py        # Ultimate program (with XOR offsets)
│   │   ├── basic_block.py     # Basic block generation
│   │   ├── memory_manager.py  # Memory allocation
│   │   └── register_fsm.py    # Register state machine
│   ├── execution/
│   │   ├── iss_runner.py      # Spike ISS interface (PROBLEM HERE)
│   │   ├── rtl_runner.py      # Verilator RTL interface
│   │   └── elf_writer.py      # ELF file generation
│   └── isa/
│       ├── instructions.py    # RISC-V instruction definitions
│       └── encoding.py        # Instruction encoding
├── deps/
│   ├── picorv32/              # PicoRV32 RTL model
│   │   └── testbench_verilator # Pre-built Verilator binary (works)
│   └── riscv-isa-sim/         # Spike source
└── output/                    # Fuzzing output directory
```

## Key Files

### `cascade/execution/iss_runner.py`
- `ISSRunner.run()`: Runs intermediate program on Spike
- `_run_spike_with_early_exit()`: Streams Spike output, looks for "ebreak" or "breakpoint"
- `_build_command()`: Builds Spike command line
- Timeout: 10 seconds by default (`config.iss_timeout = 10000ms`)

Spike command format:
```bash
/opt/riscv/bin/spike --isa rv32im -m0x80000000:0x100000 -l --log-commits program.elf
```

### `cascade/execution/elf_writer.py`
- `ELFWriter.write()`: Converts IntermediateProgram to ELF
- Entry point: `program.entry_addr`
- Load address: `program.code_start` (0x80000000)

### `cascade/generator/intermediate.py`
- `IntermediateProgram`: Contains basic blocks, CF markers, entry address
- `to_bytes()`: Serializes program to raw bytes

### `cascade/fuzzer.py`
- `_fuzz_iteration()`: Main iteration - generate → ISS → ultimate → RTL
- `_worker_fuzz()`: Worker process function

## Investigation Status

### What We Know
1. Spike is installed at `/opt/riscv/bin/spike` and works (`spike --help` succeeds)
2. RTL model is found at `deps/picorv32/testbench_verilator` (works)
3. Generated programs are valid ELF files
4. ISS executions can time out (default 10 second timeout)
5. RISC-V toolchain (assembler/linker) is NOT installed - only Spike binaries

### What We Don't Know
1. Do generated programs contain `ebreak` instruction?
2. What does Spike actually output before timing out?
3. Is the memory configuration correct?
4. Is the ISA string correct for the generated programs?

## Next Steps to Debug

### 1. Verify Programs Have ebreak
Look at `basic_block.py` and verify the intermediate program construction adds `ebreak` at the end.

### 2. Capture Spike Output
Modify ISS runner to save raw Spike output even on timeout:
```python
# In _run_spike_with_early_exit(), save output to file for debugging
with open('/tmp/spike_debug.log', 'w') as f:
    f.write(result.raw_output)
```

### 3. Test Spike Manually
Create a minimal test program and run Spike directly:
```bash
# Need to create test.elf with: nop; nop; ebreak
spike --isa rv32im -m0x80000000:0x100000 -l test.elf
```

Problem: No RISC-V assembler available to create test.elf

### 4. Check Generated Program Content
Add debug logging to dump the intermediate program bytes:
```python
# In fuzzer.py fuzz_one()
import binascii
print(f"Program bytes: {binascii.hexlify(intermediate.to_bytes())[:100]}")
```

### 5. Verify ELF Structure
Use `readelf` on a generated ELF to verify entry point and load address are correct.

## Recent Code Changes

- Progress output now uses a single overall bar (no per-worker bars).
- RTL detection accounts for PicoRV32's `testbench_verilator` binary.

## Configuration

Key config values from `cascade/config.py`:
- `cpu.xlen`: 32 (PicoRV32 is rv32)
- `cpu.extensions`: ['I', 'M'] (integer + multiply)
- `memory.code_start`: 0x80000000
- `iss_timeout`: 10000 (10 seconds)
- `rtl_timeout`: 100000 (100 seconds)
- `spike_path`: /opt/riscv/bin/spike
- `rtl_model_path`: deps/picorv32

## Running the Fuzzer

```bash
cd /home/vvdveen/cascade
source .venv/bin/activate
cascade -n 100 --cpu picorv32 --rtl-path deps/picorv32 -o output/test
```

Or programmatically:
```python
from cascade.config import FuzzerConfig, PICORV32_CONFIG
from cascade.fuzzer import Fuzzer

config = FuzzerConfig(cpu=PICORV32_CONFIG, num_programs=100)
fuzzer = Fuzzer(config)
fuzzer.calibrate()
bugs = fuzzer.fuzz()
```

## Root Cause Hypothesis

### Verified: Programs DO Have ebreak

The code flow in `intermediate.py`:
1. `generate()` creates initial block, N fuzzing blocks, final block
2. `_generate_final_block()` calls `block_gen.generate_final_block()`
3. `generate_final_block()` in `basic_block.py:488` sets `block.terminator = EncodedInstruction(EBREAK)`
4. `_fixup_control_flow()` makes each block JAL to the next block
5. Final block has ebreak, not JAL, so it doesn't get modified

**Control flow is not reliably correct - programs MAY fail to reach ebreak.**

### Root Cause Found: JALR/Branch Targets Not Set Up

**Debug output confirms the issue:**

```
Entry addr: 0x80000000
Num blocks: 96
Found EBREAK at 0x80001244

Running Spike...
TIMEOUT!
Partial stderr (66035203 chars): ...  # 66 MB of execution log!
```

The program:
1. **Does have ebreak** at 0x80001244
2. **Executes correctly** starting at 0x80000000
3. **Produces 66 MB of log** before timeout - stuck in loop
4. **Never reaches ebreak** because JALR jumps to wrong address

**The bug is in `_fixup_control_flow()` for JALR instructions:**

```python
elif term.instruction.name == 'jalr':
    # For intermediate program, make JALR jump to next block
    # by setting up the register appropriately
    if i + 1 < len(blocks):
        next_block = blocks[i + 1]
        block.jump_target_addr = next_block.start_addr

        # Update the marker with target value
        for marker in block.cf_markers:
            if marker.pc == term_pc:
                marker.target_value = next_block.start_addr
                marker.branch_target = next_block.start_addr
```

**Problem:** This sets `marker.target_value` but **doesn't actually set up the register** to contain that value. JALR and branches read registers that contain arbitrary values, so control flow can jump to wrong addresses.

**Why this is wrong:**
- JALR: `rd = pc + 4; pc = (rs1 + imm) & ~1`
- The target depends on the register value
- For intermediate programs, rs1 has whatever value the previous instructions left
- The marker records what we WANT, but the actual execution jumps somewhere else

**Why this matters:**
- ISS needs deterministic control flow to execute the intermediate program
- Without correct register setup, JALR jumps to wrong address
- This creates an infinite loop, never reaching ebreak

### How to Fix

The intermediate program needs to ensure JALR register values are correct. Options:

1. **Replace JALR with JAL** in intermediate programs (no register dependency)
2. **Insert setup instructions** before JALR to load the correct target into the register
3. **Avoid JALR entirely** in intermediate programs, only use in ultimate programs

Per the paper, the intermediate program should have **deterministic control flow** where all targets are known. JALR (indirect jump) is CF-ambiguous and should only be used in the ultimate program after ISS feedback provides the register values.

### Scope of the Problem

The `_generate_hopping_instruction()` in `basic_block.py:242` randomly chooses:
- `'jal'` (33%) - Works correctly, no register dependency
- `'jalr'` (33%) - **BROKEN** - rs1 register has random value
- `'branch'` (33%) - **BROKEN** - rs1/rs2 may not satisfy the branch condition

For a program with 96 blocks (like our test), ~32 blocks use JALR, ~32 use branches. Any broken control flow creates an infinite loop.

### Fix Options

**Option 1: Replace all terminators with JAL in intermediate programs** (Simple, safe)

In `intermediate.py:_fixup_control_flow()`, replace JALR with JAL:

```python
elif term.instruction.name == 'jalr':
    # For intermediate program, replace JALR with JAL
    if i + 1 < len(blocks):
        next_block = blocks[i + 1]
        offset = next_block.start_addr - term_pc
        block.terminator = EncodedInstruction(JAL, rd=term.rd, imm=offset)
        block.jump_target_addr = next_block.start_addr
```

For branches, replace with unconditional JAL or ensure they fall through:

```python
elif term.instruction.category.name == 'BRANCH':
    # For intermediate program, replace branch with JAL to next block
    if i + 1 < len(blocks):
        next_block = blocks[i + 1]
        offset = next_block.start_addr - term_pc
        block.terminator = EncodedInstruction(JAL, rd=0, imm=offset)
        block.jump_target_addr = next_block.start_addr
```

**Option 2: Only use JAL for intermediate block terminators** (Better design)

Modify `basic_block.py:_generate_hopping_instruction()`:

```python
def _generate_hopping_instruction(self, pc: int) -> Tuple[EncodedInstruction,
                                                           Optional[CFAmbiguousMarker]]:
    """Generate a control-flow changing instruction."""
    # For intermediate programs, only use JAL (deterministic)
    # JALR and branches are CF-ambiguous and need ISS feedback
    return self._generate_jal_instruction(pc)
```

Then in ultimate program construction, replace some JALs with JALR/branches using ISS feedback.

**Option 3: Insert register setup before JALR** (Complex, matches paper)

Before each JALR in intermediate program, insert:
```assembly
lui  rs1, target[31:12]
addi rs1, rs1, target[11:0]
jalr rd, rs1, 0
```

This is what the ultimate program does with XOR offsets, but for intermediate we'd use direct values.

## Debugging Script

Create this script to debug Spike directly:

```python
#!/usr/bin/env python3
"""Debug script to test Spike with generated programs."""
import tempfile
import subprocess
from pathlib import Path

# Add cascade to path
import sys
sys.path.insert(0, '/home/vvdveen/cascade')

from cascade.config import FuzzerConfig, PICORV32_CONFIG
from cascade.generator.intermediate import IntermediateProgramGenerator
from cascade.execution.elf_writer import ELFWriter

# Create config
config = FuzzerConfig(cpu=PICORV32_CONFIG)

# Generate one intermediate program
generator = IntermediateProgramGenerator(config)
program = generator.generate(seed=12345)

# Write to ELF
with tempfile.TemporaryDirectory() as tmpdir:
    elf_path = Path(tmpdir) / "test.elf"
    writer = ELFWriter(config.cpu.xlen)
    writer.write(program, elf_path)

    # Print program info
    print(f"Entry addr: {hex(program.entry_addr)}")
    print(f"Code start: {hex(program.code_start)}")
    print(f"Num blocks: {len(program.blocks)}")
    print(f"Code bytes: {len(program.to_bytes())} bytes")

    # Print first few instructions
    code = program.to_bytes()
    for i in range(0, min(40, len(code)), 4):
        word = int.from_bytes(code[i:i+4], 'little')
        print(f"  {hex(program.code_start + i)}: {word:08x}")

    # Build Spike command
    cmd = [
        '/opt/riscv/bin/spike',
        '--isa', 'rv32im',
        '-m0x80000000:0x100000',
        '-l', '--log-commits',
        str(elf_path)
    ]
    print(f"\nCommand: {' '.join(cmd)}")

    # Run Spike with short timeout
    print("\nRunning Spike...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"Return code: {result.returncode}")
        print(f"Stdout ({len(result.stdout)} chars):")
        print(result.stdout[:2000])
        print(f"\nStderr ({len(result.stderr)} chars):")
        print(result.stderr[:2000])
    except subprocess.TimeoutExpired as e:
        print("TIMEOUT!")
        stdout = e.stdout.decode() if e.stdout else ""
        stderr = e.stderr.decode() if e.stderr else ""
        print(f"Partial stdout: {stdout[:1000]}")
        print(f"Partial stderr: {stderr[:1000]}")
```

Run with:
```bash
cd /home/vvdveen/cascade
source .venv/bin/activate
python3 debug_spike.py
```

## Files to Read

- `cascade/generator/basic_block.py` - How are programs terminated?
- `cascade/fuzzer.py:260-310` - The `fuzz_one()` function
- `cascade/execution/iss_runner.py:145-210` - The `_run_spike_with_early_exit()` function
