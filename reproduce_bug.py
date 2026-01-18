import subprocess
import sys
import shutil
from pathlib import Path

def run_command(cmd, timeout=10):
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return proc
    except subprocess.TimeoutExpired as e:
        # Return a dummy object with the captured output
        class TimeoutProc:
            returncode = -1
            stdout = e.stdout if e.stdout else ""
            stderr = e.stderr if e.stderr else ""
        return TimeoutProc()

def reproduce():
    bug_dir = Path("output/reduced_bugs/bug_20260117_171416_81")
    elf_path = bug_dir / "reduced.elf"
    hex_path = bug_dir / "reduced.hex"

    if not elf_path.exists():
        print(f"Error: {elf_path} not found.")
        sys.exit(1)

    print(f"Reproducing bug from: {elf_path}")

    # 1. Create HEX file for PicoRV32
    print("Converting ELF to HEX...")
    
    # Manual ELF parsing to extract code segment (Loadable segment)
    with open(elf_path, "rb") as f:
        elf_data = f.read()
        
    # Check magic
    if elf_data[0:4] != b'\x7fELF':
        print("Not an ELF file")
        sys.exit(1)
        
    # Parse e_phoff (Program Header Offset) - 32-bit ELF
    # e_phoff is at offset 28, 4 bytes, little endian
    e_entry = int.from_bytes(elf_data[24:28], 'little')
    print(f"ELF Entry Point: 0x{e_entry:08x}")
    e_phoff = int.from_bytes(elf_data[28:32], 'little')
    e_phnum = int.from_bytes(elf_data[44:46], 'little')
    e_phentsize = int.from_bytes(elf_data[42:44], 'little')
    
    code_data = b""
    
    # Iterate Program Headers to find PT_LOAD (1)
    for i in range(e_phnum):
        ph_offset = e_phoff + i * e_phentsize
        p_type = int.from_bytes(elf_data[ph_offset:ph_offset+4], 'little')
        
        if p_type == 1: # PT_LOAD
            p_offset = int.from_bytes(elf_data[ph_offset+4:ph_offset+8], 'little')
            p_filesz = int.from_bytes(elf_data[ph_offset+16:ph_offset+20], 'little')
            
            # Extract data
            code_data = elf_data[p_offset : p_offset + p_filesz]
            break
            
    if not code_data:
        print("Error: Could not find PT_LOAD segment in ELF")
        sys.exit(1)
        
    # Now format as hex for Verilog readmemh
    with open(hex_path, "w") as f_hex:
        # PicoRV32 testbench expects @0 start
        f_hex.write("@00000000\n")
        
        # Write 32-bit words (little endian from binary to hex)
        # Pad with zeros if needed
        data = code_data
        for i in range(0, len(data), 4):
            chunk = data[i:i+4]
            if len(chunk) < 4:
                chunk = chunk + b'\x00' * (4 - len(chunk))
            
            val = int.from_bytes(chunk, byteorder='little')
            f_hex.write(f"{val:08x}\n")
            
    print(f"Created {hex_path}")

    # 2. Run on PicoRV32 RTL
    print("\n--- Running on PicoRV32 RTL ---")
    sim_binary = Path("deps/picorv32/testbench_verilator_dir/Vpicorv32_wrapper")
    if not sim_binary.exists():
        print(f"Error: Simulator binary {sim_binary} not found.")
        sys.exit(1)

    cmd_rtl = [
        str(sim_binary),
        f"+firmware={hex_path}",
        "+max_cycles=10000"
    ]
    
    proc_rtl = run_command(cmd_rtl)
    
    if proc_rtl is None:
        print("RESULT: TIMEOUT (Bug Triggered!)")
        print("The processor hung or entered an infinite loop.")
    elif proc_rtl.returncode != 0:
         print(f"RESULT: FAILED (Exit Code {proc_rtl.returncode})")
         print("Output snippet:")
         print(proc_rtl.stdout[:500])
         print(proc_rtl.stderr[:500])
    else:
        print("RESULT: SUCCESS (No Bug?)")
        # Check for TRAP which indicates success in this testbench
        if "TRAP" in proc_rtl.stdout:
            print("Output contains TRAP (Normal termination)")
        else:
            print("Output snippet:")
            print(proc_rtl.stdout[:500])

    # 3. Run on Spike ISS (Reference)
    print("\n--- Running on Spike ISS (Reference) ---")
    spike_bin = "/opt/riscv/bin/spike"
    
    # -l for log, --isa for architecture
    cmd_iss = [
        spike_bin,
        "-l", "--log-commits",
        "--isa", "rv32im",
        "-m0x80000000:0x100000",
        str(elf_path)
    ]
    
    proc_iss = run_command(cmd_iss)
    
    if proc_iss is None:
        print("RESULT: TIMEOUT")
        print("Note: Spike timed out, which is expected for the reduced loop.")
    elif proc_iss.returncode != 0:
        # Spike returns non-zero on trap/breakpoint sometimes depending on how it exits
        # But for us, "Success" means it ran instructions and finished.
        # Actually, if it hits the "bug", it would diverge or crash.
        print(f"RESULT: Exited with {proc_iss.returncode}")
        print("Output snippet (tail):")
        print(proc_iss.stderr[-1000:]) # Print tail to see the loop
    else:
        print("RESULT: SUCCESS")
        print("The reference model executed the program correctly.")


if __name__ == "__main__":
    reproduce()
