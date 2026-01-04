"""Tests for input format API contract for ML test functions.

These tests verify that ML functions accept mixed-type dictionary input
including integers, strings (categorical), and callable datasets.
"""

import numbers

import pytest

from surfaces.test_functions import KNeighborsClassifierFunction
from surfaces.test_functions.machine_learning.tabular.classification.datasets import iris_data


def is_numeric(value):
    """Check if value is a numeric type (int, float, numpy scalar)."""
    return isinstance(value, numbers.Number)


@pytest.mark.ml
class TestMLFunctionInput:
    """Test ML functions accept mixed-type dictionary input."""

    def test_dict_with_mixed_types(self):
        """Test ML function with int, string, and callable parameters."""
        func = KNeighborsClassifierFunction()
        result = func(
            {
                "n_neighbors": 5,
                "algorithm": "auto",
                "cv": 3,
                "dataset": iris_data,
            }
        )
        assert is_numeric(result)

    def test_dict_with_string_categorical(self):
        """Test ML function with different categorical string values."""
        func = KNeighborsClassifierFunction()
        for algo in ["auto", "ball_tree", "kd_tree", "brute"]:
            result = func(
                {
                    "n_neighbors": 5,
                    "algorithm": algo,
                    "cv": 3,
                    "dataset": iris_data,
                }
            )
            assert is_numeric(result)
