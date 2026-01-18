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

    # Print first and last few instructions
    code = program.to_bytes()
    print("\nFirst 10 instructions:")
    for i in range(0, min(40, len(code)), 4):
        word = int.from_bytes(code[i:i+4], 'little')
        print(f"  {hex(program.code_start + i)}: {word:08x}")

    print(f"\nLast 10 instructions:")
    start = max(0, len(code) - 40)
    start = (start // 4) * 4  # Align to 4
    for i in range(start, len(code), 4):
        word = int.from_bytes(code[i:i+4], 'little')
        print(f"  {hex(program.code_start + i)}: {word:08x}")

    # Check for ebreak (0x00100073)
    ebreak_encoding = 0x00100073
    found_ebreak = False
    for i in range(0, len(code), 4):
        word = int.from_bytes(code[i:i+4], 'little')
        if word == ebreak_encoding:
            print(f"\nFound EBREAK at {hex(program.code_start + i)}")
            found_ebreak = True
    if not found_ebreak:
        print("\nWARNING: No EBREAK instruction found!")

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
        print(f"Partial stdout ({len(stdout)} chars): {stdout[:1000]}")
        print(f"Partial stderr ({len(stderr)} chars): {stderr[:1000]}")
