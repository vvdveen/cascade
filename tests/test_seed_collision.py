"""
Tests for seed collision detection.
"""

import pytest

from cascade.config import FuzzerConfig, CPUConfig, Extension
from cascade.fuzzer import Fuzzer, SeedCollisionError


def test_seed_collision_detected_within_run(tmp_path):
    config = FuzzerConfig(
        cpu=CPUConfig(name="test", xlen=32, extensions={Extension.I}),
        output_dir=tmp_path,
    )
    fuzzer = Fuzzer(config)

    fuzzer._record_seed(123)
    with pytest.raises(SeedCollisionError):
        fuzzer._record_seed(123)


def test_seed_collision_detected_across_runs(tmp_path):
    config = FuzzerConfig(
        cpu=CPUConfig(name="test", xlen=32, extensions={Extension.I}),
        output_dir=tmp_path,
    )
    fuzzer = Fuzzer(config)
    fuzzer._record_seed(456)

    fuzzer2 = Fuzzer(config)
    with pytest.raises(SeedCollisionError):
        fuzzer2._record_seed(456)
