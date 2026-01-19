"""
Integration tests that run the fuzzer with real ISS/RTL binaries.
"""

import os
from pathlib import Path

import pytest

from cascade.config import (
    FuzzerConfig, PICORV32_CONFIG, KRONOS_CONFIG, KRONOS_MEMORY_LAYOUT
)
from cascade.fuzzer import Fuzzer
from cascade.execution.rtl_runner import RTLRunner


def _find_spike() -> Path | None:
    candidates = [
        os.environ.get("SPIKE_PATH"),
        str(Path.home() / ".local" / "riscv" / "bin" / "spike"),
        "/opt/riscv/bin/spike",
        "/opt/homebrew/bin/spike",
    ]
    for cand in candidates:
        if not cand:
            continue
        path = Path(cand)
        if path.exists():
            return path
    return None


def _require_rtl_binary(cpu_name: str, rtl_path: Path) -> None:
    config = FuzzerConfig(cpu=PICORV32_CONFIG, rtl_model_path=rtl_path)
    if cpu_name == "kronos":
        config.cpu = KRONOS_CONFIG
        config.memory = KRONOS_MEMORY_LAYOUT
    runner = RTLRunner(config)
    if runner._get_sim_binary() is None:
        pytest.skip(f"RTL binary not found for {cpu_name} at {rtl_path}")


@pytest.mark.integration
def test_fuzzer_runs_picorv32(tmp_path):
    spike = _find_spike()
    if spike is None:
        pytest.skip("Spike not found")
    rtl_path = Path("deps/picorv32")
    _require_rtl_binary("picorv32", rtl_path)

    config = FuzzerConfig(
        cpu=PICORV32_CONFIG,
        spike_path=spike,
        rtl_model_path=rtl_path,
        num_programs=1,
        num_workers=1,
        output_dir=tmp_path / "picorv32",
    )
    fuzzer = Fuzzer(config)
    assert fuzzer.calibrate() is True
    fuzzer.fuzz(num_programs=1)


@pytest.mark.integration
def test_fuzzer_runs_kronos(tmp_path):
    spike = _find_spike()
    if spike is None:
        pytest.skip("Spike not found")
    rtl_path = Path("deps/kronos")
    _require_rtl_binary("kronos", rtl_path)

    config = FuzzerConfig(
        cpu=KRONOS_CONFIG,
        memory=KRONOS_MEMORY_LAYOUT,
        spike_path=spike,
        rtl_model_path=rtl_path,
        num_programs=1,
        num_workers=1,
        output_dir=tmp_path / "kronos",
    )
    fuzzer = Fuzzer(config)
    assert fuzzer.calibrate() is True
    fuzzer.fuzz(num_programs=1)
