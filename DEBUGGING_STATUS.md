# Cascade Debugging Status

This document brings another LLM or developer up to speed with the current debugging state of the Cascade CPU fuzzer.

## What is Cascade?

Cascade is a RISC-V CPU fuzzer based on the USENIX Security 2024 paper "Cascade: CPU Fuzzing via Intricate Program Generation" by Solt et al. The key innovation is **Asymmetric ISA Pre-simulation**:

1. **Intermediate Program**: Generate a program with CF-ambiguous instructions using placeholder values
2. **ISS Execution**: Run the intermediate program on Spike (RISC-V ISS) to collect register values
3. **Ultimate Program**: Use ISS feedback to construct XOR-based offsets that make control flow deterministic
4. **RTL Execution**: Run the ultimate program on RTL (Verilator). A timeout indicates a bug.

The ISS is NOT used for differential comparison - it's used to *construct* valid programs. Programs terminate with `ebreak` instruction. Bug detection is timeout-based (program should complete, non-termination = bug).

## Current Problem

**The fuzzer doesn't work because Spike (ISS) times out on every generated program.**

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

Without ISS feedback, the fuzzer cannot:
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
- Timeout: 20 seconds (`config.iss_timeout = 20000ms`)

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
- `Fuzzer.fuzz_one()`: Main iteration - generate → ISS → ultimate → RTL
- `_worker_fuzz()`: Worker process function
- Line 297: `logger.warning(f"ISS failed at iteration {iteration}: {iss_result.error_message}")`

## Investigation Status

### What We Know
1. Spike is installed at `/opt/riscv/bin/spike` and works (`spike --help` succeeds)
2. RTL model is found at `deps/picorv32/testbench_verilator` (works)
3. Generated programs are valid ELF files
4. Every ISS execution times out (20 second timeout)
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

### Progress Bar (fixed)
Changed from `\r` carriage return to newline for file-friendly output.

### RTL Model Detection (fixed)
Fixed logic to check for pre-built Verilator binary before checking if Verilator is installed.

### Worker Logging (added)
Each worker now writes to `output/worker_N.log` for debugging.

### Progress Updates (improved)
Only print progress when `completed > last_completed`.

## Configuration

Key config values from `cascade/config.py`:
- `cpu.xlen`: 32 (PicoRV32 is rv32)
- `cpu.extensions`: ['I', 'M'] (integer + multiply)
- `memory.code_start`: 0x80000000
- `iss_timeout`: 20000 (20 seconds)
- `rtl_timeout`: 20000 (20 seconds)
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

**Control flow is correct - programs SHOULD reach ebreak.**

### Remaining Hypotheses

Since programs have ebreak and control flow looks correct, the issue is likely:

1. **ELF Structure Problem**: Entry point or load address mismatch
   - Entry point should be `code_start` (0x80000000)
   - Load address should match
   - Verify with `readelf -h program.elf`

2. **Memory Configuration for Spike**:
   - Command: `-m0x80000000:0x100000`
   - This sets memory at 0x80000000 with size 1MB
   - Might need different format or additional flags

3. **Spike Output Detection**:
   - Code looks for "ebreak" or "breakpoint" in output
   - Spike's commit log might use different wording
   - Need to capture actual Spike output to verify

4. **Initial Block Problem**:
   - `generate_initial_block()` might have issues
   - Might jump somewhere unexpected before reaching ebreak

5. **Instruction Encoding Bug**:
   - EBREAK might be encoded incorrectly
   - JAL offsets might be calculated wrong

The `_is_expected_termination()` function looks for "ebreak" or "breakpoint" in Spike output. If the output uses different wording, it won't detect termination.

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
