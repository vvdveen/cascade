"""
Execution module for Cascade.

Handles running programs on ISS (Spike) and RTL simulators (Verilator).
"""

from .elf_writer import ELFWriter
from .iss_runner import ISSRunner, ISSResult
from .rtl_runner import RTLRunner, RTLResult
