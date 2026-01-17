"""
Tests for reduction utilities.
"""

from cascade.reduction.reducer import ReductionResult


def test_reduction_result_default_reduced_size():
    """ReductionResult should allow construction with only original_size."""
    result = ReductionResult(original_size=10)

    assert result.reduced_size == 0
