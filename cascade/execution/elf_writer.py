"""
ELF file generation for RISC-V programs.

Generates minimal ELF binaries suitable for bare-metal execution
on RISC-V simulators and hardware.
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ..generator.intermediate import IntermediateProgram
from ..generator.ultimate import UltimateProgram


# ELF Constants
ELF_MAGIC = b'\x7fELF'

# ELF Class
ELFCLASS32 = 1
ELFCLASS64 = 2

# ELF Data encoding
ELFDATA2LSB = 1  # Little-endian

# ELF Type
ET_EXEC = 2  # Executable

# ELF Machine
EM_RISCV = 243

# Program header types
PT_LOAD = 1

# Program header flags
PF_X = 0x1  # Execute
PF_W = 0x2  # Write
PF_R = 0x4  # Read


@dataclass
class ELFHeader:
    """ELF file header."""
    ei_class: int = ELFCLASS32
    ei_data: int = ELFDATA2LSB
    e_type: int = ET_EXEC
    e_machine: int = EM_RISCV
    e_version: int = 1
    e_entry: int = 0
    e_phoff: int = 0
    e_shoff: int = 0
    e_flags: int = 0
    e_ehsize: int = 0
    e_phentsize: int = 0
    e_phnum: int = 0
    e_shentsize: int = 0
    e_shnum: int = 0
    e_shstrndx: int = 0


@dataclass
class ProgramHeader:
    """ELF program header."""
    p_type: int = PT_LOAD
    p_offset: int = 0
    p_vaddr: int = 0
    p_paddr: int = 0
    p_filesz: int = 0
    p_memsz: int = 0
    p_flags: int = PF_R | PF_X
    p_align: int = 0x1000


class ELFWriter:
    """
    Writes RISC-V programs to ELF format.

    Supports both RV32 and RV64.
    """

    def __init__(self, xlen: int = 32):
        """
        Initialize ELF writer.

        Args:
            xlen: Register width (32 or 64)
        """
        self.xlen = xlen
        self.ei_class = ELFCLASS32 if xlen == 32 else ELFCLASS64

    def write(self, program: Union[IntermediateProgram, UltimateProgram],
              output_path: Path) -> None:
        """
        Write program to ELF file.

        Args:
            program: Program to write
            output_path: Output file path
        """
        elf_data = self.to_bytes(program)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(elf_data)

    def to_bytes(self, program: Union[IntermediateProgram, UltimateProgram]) -> bytes:
        """
        Convert program to ELF bytes.

        Args:
            program: Program to convert

        Returns:
            ELF file as bytes
        """
        # Get program data
        code_data = program.to_bytes()
        entry_addr = program.entry_addr
        code_start = program.code_start

        if self.xlen == 32:
            return self._build_elf32(code_data, entry_addr, code_start)
        else:
            return self._build_elf64(code_data, entry_addr, code_start)

    def _build_elf32(self, code: bytes, entry: int, load_addr: int) -> bytes:
        """Build 32-bit ELF file."""
        # ELF header size
        ehdr_size = 52
        phdr_size = 32
        shdr_size = 40  # Section header size for 32-bit

        # We have one program header (for code)
        num_phdrs = 1
        num_shdrs = 1  # Just the null section header (required by some loaders)

        # Calculate offsets
        phdr_offset = ehdr_size
        shdr_offset = ehdr_size + (phdr_size * num_phdrs)
        code_offset = shdr_offset + (shdr_size * num_shdrs)

        # Align code to page boundary
        code_offset = (code_offset + 0xFFF) & ~0xFFF

        # Build ELF header
        ehdr = self._build_elf32_header(
            entry=entry,
            phoff=phdr_offset,
            phnum=num_phdrs,
            shoff=shdr_offset,
            shnum=num_shdrs,
            shstrndx=0  # SHN_UNDEF - no section name string table
        )

        # Build program header for code
        phdr = self._build_phdr32(
            p_type=PT_LOAD,
            p_offset=code_offset,
            p_vaddr=load_addr,
            p_paddr=load_addr,
            p_filesz=len(code),
            p_memsz=len(code),
            p_flags=PF_R | PF_X,
            p_align=0x1000
        )

        # Build null section header (required as first entry)
        shdr_null = self._build_shdr32_null()

        # Assemble ELF
        elf = bytearray()
        elf.extend(ehdr)
        elf.extend(phdr)
        elf.extend(shdr_null)

        # Pad to code offset
        elf.extend(b'\x00' * (code_offset - len(elf)))

        # Add code
        elf.extend(code)

        return bytes(elf)

    def _build_shdr32_null(self) -> bytes:
        """Build null section header (all zeros, 40 bytes)."""
        return b'\x00' * 40

    def _build_elf64(self, code: bytes, entry: int, load_addr: int) -> bytes:
        """Build 64-bit ELF file."""
        # ELF header size
        ehdr_size = 64
        phdr_size = 56

        num_phdrs = 1
        phdr_offset = ehdr_size
        code_offset = ehdr_size + (phdr_size * num_phdrs)
        code_offset = (code_offset + 0xFFF) & ~0xFFF

        ehdr = self._build_elf64_header(
            entry=entry,
            phoff=phdr_offset,
            phnum=num_phdrs
        )

        phdr = self._build_phdr64(
            p_type=PT_LOAD,
            p_offset=code_offset,
            p_vaddr=load_addr,
            p_paddr=load_addr,
            p_filesz=len(code),
            p_memsz=len(code),
            p_flags=PF_R | PF_X,
            p_align=0x1000
        )

        elf = bytearray()
        elf.extend(ehdr)
        elf.extend(phdr)
        elf.extend(b'\x00' * (code_offset - len(elf)))
        elf.extend(code)

        return bytes(elf)

    def _build_elf32_header(self, entry: int, phoff: int, phnum: int,
                             shoff: int = 0, shnum: int = 0, shstrndx: int = 0) -> bytes:
        """Build 32-bit ELF header."""
        ehdr = bytearray()

        # e_ident (16 bytes)
        ehdr.extend(ELF_MAGIC)           # Magic
        ehdr.append(ELFCLASS32)          # 32-bit
        ehdr.append(ELFDATA2LSB)         # Little-endian
        ehdr.append(1)                   # EV_CURRENT
        ehdr.append(0)                   # ELFOSABI_NONE
        ehdr.extend(b'\x00' * 8)         # Padding

        # e_type (2 bytes)
        ehdr.extend(struct.pack('<H', ET_EXEC))

        # e_machine (2 bytes)
        ehdr.extend(struct.pack('<H', EM_RISCV))

        # e_version (4 bytes)
        ehdr.extend(struct.pack('<I', 1))

        # e_entry (4 bytes)
        ehdr.extend(struct.pack('<I', entry))

        # e_phoff (4 bytes)
        ehdr.extend(struct.pack('<I', phoff))

        # e_shoff (4 bytes)
        ehdr.extend(struct.pack('<I', shoff))

        # e_flags (4 bytes) - RVC/RVE flags could go here
        ehdr.extend(struct.pack('<I', 0))

        # e_ehsize (2 bytes)
        ehdr.extend(struct.pack('<H', 52))

        # e_phentsize (2 bytes)
        ehdr.extend(struct.pack('<H', 32))

        # e_phnum (2 bytes)
        ehdr.extend(struct.pack('<H', phnum))

        # e_shentsize (2 bytes)
        ehdr.extend(struct.pack('<H', 40 if shnum > 0 else 0))

        # e_shnum (2 bytes)
        ehdr.extend(struct.pack('<H', shnum))

        # e_shstrndx (2 bytes)
        ehdr.extend(struct.pack('<H', shstrndx))

        return bytes(ehdr)

    def _build_elf64_header(self, entry: int, phoff: int, phnum: int) -> bytes:
        """Build 64-bit ELF header."""
        ehdr = bytearray()

        # e_ident (16 bytes)
        ehdr.extend(ELF_MAGIC)
        ehdr.append(ELFCLASS64)
        ehdr.append(ELFDATA2LSB)
        ehdr.append(1)
        ehdr.append(0)
        ehdr.extend(b'\x00' * 8)

        # e_type
        ehdr.extend(struct.pack('<H', ET_EXEC))

        # e_machine
        ehdr.extend(struct.pack('<H', EM_RISCV))

        # e_version
        ehdr.extend(struct.pack('<I', 1))

        # e_entry (8 bytes)
        ehdr.extend(struct.pack('<Q', entry))

        # e_phoff (8 bytes)
        ehdr.extend(struct.pack('<Q', phoff))

        # e_shoff (8 bytes)
        ehdr.extend(struct.pack('<Q', 0))

        # e_flags
        ehdr.extend(struct.pack('<I', 0))

        # e_ehsize
        ehdr.extend(struct.pack('<H', 64))

        # e_phentsize
        ehdr.extend(struct.pack('<H', 56))

        # e_phnum
        ehdr.extend(struct.pack('<H', phnum))

        # e_shentsize
        ehdr.extend(struct.pack('<H', 0))

        # e_shnum
        ehdr.extend(struct.pack('<H', 0))

        # e_shstrndx
        ehdr.extend(struct.pack('<H', 0))

        return bytes(ehdr)

    def _build_phdr32(self, p_type: int, p_offset: int, p_vaddr: int,
                      p_paddr: int, p_filesz: int, p_memsz: int,
                      p_flags: int, p_align: int) -> bytes:
        """Build 32-bit program header."""
        return struct.pack('<IIIIIIII',
                           p_type,
                           p_offset,
                           p_vaddr,
                           p_paddr,
                           p_filesz,
                           p_memsz,
                           p_flags,
                           p_align)

    def _build_phdr64(self, p_type: int, p_offset: int, p_vaddr: int,
                      p_paddr: int, p_filesz: int, p_memsz: int,
                      p_flags: int, p_align: int) -> bytes:
        """Build 64-bit program header."""
        return struct.pack('<IIQQQQQQ',
                           p_type,
                           p_flags,
                           p_offset,
                           p_vaddr,
                           p_paddr,
                           p_filesz,
                           p_memsz,
                           p_align)


def write_raw_binary(program: Union[IntermediateProgram, UltimateProgram],
                     output_path: Path) -> None:
    """
    Write program as raw binary (no ELF headers).

    Useful for some simulators that expect raw binaries.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(program.to_bytes())


def write_hex(program: Union[IntermediateProgram, UltimateProgram],
              output_path: Path, words_per_line: int = 1) -> None:
    """
    Write program as hex file (for Verilog $readmemh).

    Args:
        program: Program to write
        output_path: Output file path
        words_per_line: Number of 32-bit words per line
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = program.to_bytes()

    with open(output_path, 'w') as f:
        # Write address at start
        f.write(f"@{program.code_start >> 2:08X}\n")

        # Write data
        for i in range(0, len(data), 4 * words_per_line):
            words = []
            for j in range(words_per_line):
                offset = i + j * 4
                if offset + 4 <= len(data):
                    word = int.from_bytes(data[offset:offset + 4], 'little')
                    words.append(f"{word:08X}")
            if words:
                f.write(' '.join(words) + '\n')
