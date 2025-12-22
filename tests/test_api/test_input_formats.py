"""
Tests that define the API contract for input formats accepted by test functions.

These tests serve as documentation of what input types are supported.
If a test here fails, it means the API contract has been broken.
"""

import numbers

import numpy as np
import pytest

from surfaces.test_functions import SphereFunction, RastriginFunction
from surfaces.test_functions import KNeighborsClassifierFunction


def is_numeric(value):
    """Check if value is a numeric type (int, float, numpy scalar)."""
    return isinstance(value, numbers.Number)


# =============================================================================
# Mathematical Functions - Input Format Tests
# =============================================================================


class TestDictInput:
    """Test functions accept dictionary input with parameter names as keys."""

    def test_dict_with_float_values(self):
        func = SphereFunction(n_dim=2)
        result = func({"x0": 1.0, "x1": 2.0})
        assert is_numeric(result)

    def test_dict_with_int_values(self):
        func = SphereFunction(n_dim=2)
        result = func({"x0": 1, "x1": 2})
        assert is_numeric(result)

    def test_dict_with_numpy_scalar(self):
        func = SphereFunction(n_dim=2)
        result = func({"x0": np.float64(1.0), "x1": np.float64(2.0)})
        assert is_numeric(result)

    def test_dict_with_mixed_numeric_types(self):
        func = SphereFunction(n_dim=2)
        result = func({"x0": 1, "x1": 2.5})
        assert is_numeric(result)


class TestArrayInput:
    """Test functions accept numpy array input (values in sorted key order)."""

    def test_numpy_1d_array(self):
        func = SphereFunction(n_dim=2)
        result = func(np.array([1.0, 2.0]))
        assert is_numeric(result)

    def test_numpy_array_int_dtype(self):
        func = SphereFunction(n_dim=2)
        result = func(np.array([1, 2]))
        assert is_numeric(result)

    def test_numpy_array_float32(self):
        func = SphereFunction(n_dim=2)
        result = func(np.array([1.0, 2.0], dtype=np.float32))
        assert is_numeric(result)


class TestListInput:
    """Test functions accept list input (values in sorted key order)."""

    def test_list_of_floats(self):
        func = SphereFunction(n_dim=2)
        result = func([1.0, 2.0])
        assert is_numeric(result)

    def test_list_of_ints(self):
        func = SphereFunction(n_dim=2)
        result = func([1, 2])
        assert is_numeric(result)

    def test_list_of_mixed_types(self):
        func = SphereFunction(n_dim=2)
        result = func([1, 2.5])
        assert is_numeric(result)


class TestTupleInput:
    """Test functions accept tuple input (values in sorted key order)."""

    def test_tuple_of_floats(self):
        func = SphereFunction(n_dim=2)
        result = func((1.0, 2.0))
        assert is_numeric(result)

    def test_tuple_of_ints(self):
        func = SphereFunction(n_dim=2)
        result = func((1, 2))
        assert is_numeric(result)


class TestKwargsInput:
    """Test functions accept keyword arguments."""

    def test_kwargs_only(self):
        func = SphereFunction(n_dim=2)
        result = func(x0=1.0, x1=2.0)
        assert is_numeric(result)

    def test_dict_with_kwargs_override(self):
        func = SphereFunction(n_dim=2)
        result = func({"x0": 1.0}, x1=2.0)
        assert is_numeric(result)


class TestHigherDimensions:
    """Test that input formats work for higher dimensional functions."""

    def test_dict_5d(self):
        func = SphereFunction(n_dim=5)
        params = {f"x{i}": float(i) for i in range(5)}
        result = func(params)
        assert is_numeric(result)

    def test_array_5d(self):
        func = SphereFunction(n_dim=5)
        result = func(np.arange(5, dtype=float))
        assert is_numeric(result)

    def test_list_5d(self):
        func = SphereFunction(n_dim=5)
        result = func([0.0, 1.0, 2.0, 3.0, 4.0])
        assert is_numeric(result)


class Test1DFunction:
    """Test input formats for 1D functions."""

    def test_dict_1d(self):
        from surfaces.test_functions import GramacyAndLeeFunction

        func = GramacyAndLeeFunction()
        result = func({"x0": 0.5})
        assert is_numeric(result)

    def test_array_1d(self):
        from surfaces.test_functions import GramacyAndLeeFunction

        func = GramacyAndLeeFunction()
        result = func(np.array([0.5]))
        assert is_numeric(result)

    def test_list_1d(self):
        from surfaces.test_functions import GramacyAndLeeFunction

        func = GramacyAndLeeFunction()
        result = func([0.5])
        assert is_numeric(result)


# =============================================================================
# ML Functions - Input Format Tests (mixed types: int, str, callable)
# =============================================================================


class TestMLFunctionInput:
    """Test ML functions accept mixed-type dictionary input."""

    def test_dict_with_mixed_types(self):
        from surfaces.test_functions.machine_learning.tabular.classification.datasets import (
            iris_data,
        )

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
        from surfaces.test_functions.machine_learning.tabular.classification.datasets import (
            iris_data,
        )

        func = KNeighborsClassifierFunction()
        # Test different categorical values
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


# =============================================================================
# Consistency Tests - Same input via different formats should give same result
# =============================================================================


class TestInputConsistency:
    """Test that different input formats produce the same result."""

    def test_dict_vs_array(self):
        func = SphereFunction(n_dim=2)
        dict_result = func({"x0": 1.0, "x1": 2.0})
        array_result = func(np.array([1.0, 2.0]))
        assert dict_result == array_result

    def test_dict_vs_list(self):
        func = SphereFunction(n_dim=2)
        dict_result = func({"x0": 1.0, "x1": 2.0})
        list_result = func([1.0, 2.0])
        assert dict_result == list_result

    def test_dict_vs_tuple(self):
        func = SphereFunction(n_dim=2)
        dict_result = func({"x0": 1.0, "x1": 2.0})
        tuple_result = func((1.0, 2.0))
        assert dict_result == tuple_result

    def test_dict_vs_kwargs(self):
        func = SphereFunction(n_dim=2)
        dict_result = func({"x0": 1.0, "x1": 2.0})
        kwargs_result = func(x0=1.0, x1=2.0)
        assert dict_result == kwargs_result

    def test_all_formats_equal(self):
        """All input formats should produce identical results."""
        func = RastriginFunction(n_dim=3)
        values = [1.5, 2.5, 3.5]

        dict_result = func({"x0": values[0], "x1": values[1], "x2": values[2]})
        array_result = func(np.array(values))
        list_result = func(values)
        tuple_result = func(tuple(values))
        kwargs_result = func(x0=values[0], x1=values[1], x2=values[2])

        assert dict_result == array_result == list_result == tuple_result == kwargs_result


# =============================================================================
# Error Cases - Document expected error behavior
# =============================================================================


class TestInputErrors:
    """Test that invalid inputs raise appropriate errors."""

    def test_wrong_number_of_values_array(self):
        func = SphereFunction(n_dim=2)
        with pytest.raises(ValueError, match="Expected 2 values"):
            func(np.array([1.0, 2.0, 3.0]))

    def test_wrong_number_of_values_list(self):
        func = SphereFunction(n_dim=2)
        with pytest.raises(ValueError, match="Expected 2 values"):
            func([1.0])
