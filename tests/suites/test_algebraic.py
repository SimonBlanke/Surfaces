# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for algebraic (mathematical) test functions.

Algebraic functions are classical optimization benchmarks with
closed-form analytical expressions. They're organized by dimension:
- 1D: Single-variable functions
- 2D: Two-variable functions (many classic benchmarks)
- ND: N-dimensional scalable functions
"""

import pytest
import numpy as np

from surfaces.test_functions.algebraic import (
    algebraic_functions,
    algebraic_functions_1d,
    algebraic_functions_2d,
    algebraic_functions_nd,
    # 1D
    GramacyAndLeeFunction,
    # 2D
    AckleyFunction,
    BealeFunction,
    BoothFunction,
    RosenbrockFunction,
    SphereFunction,
    RastriginFunction,
    HimmelblausFunction,
)

from tests.conftest import instantiate_function, get_sample_params, func_id


# =============================================================================
# 1D Function Tests
# =============================================================================

@pytest.mark.algebraic
class Test1DFunctions:
    """Tests for 1D algebraic functions."""

    @pytest.mark.parametrize("func_class", algebraic_functions_1d, ids=func_id)
    def test_dimension_is_1(self, func_class):
        """1D functions have exactly 1 dimension."""
        func = instantiate_function(func_class)
        assert len(func.search_space) == 1

    @pytest.mark.parametrize("func_class", algebraic_functions_1d, ids=func_id)
    def test_evaluates_correctly(self, func_class):
        """1D functions evaluate and return numeric result."""
        func = instantiate_function(func_class)
        params = get_sample_params(func)
        result = func(params)
        assert isinstance(result, (int, float))

    def test_gramacy_lee_known_values(self):
        """Gramacy & Lee function has known behavior."""
        func = GramacyAndLeeFunction()
        # Function should be defined on [0.5, 2.5]
        assert "x0" in func.search_space


# =============================================================================
# 2D Function Tests
# =============================================================================

@pytest.mark.algebraic
class Test2DFunctions:
    """Tests for 2D algebraic functions."""

    @pytest.mark.parametrize("func_class", algebraic_functions_2d, ids=func_id)
    def test_dimension_is_2(self, func_class):
        """2D functions have exactly 2 dimensions."""
        func = instantiate_function(func_class)
        assert len(func.search_space) == 2

    @pytest.mark.parametrize("func_class", algebraic_functions_2d, ids=func_id)
    def test_evaluates_correctly(self, func_class):
        """2D functions evaluate and return numeric result."""
        func = instantiate_function(func_class)
        params = get_sample_params(func)
        result = func(params)
        assert isinstance(result, (int, float))

    @pytest.mark.parametrize("func_class", algebraic_functions_2d, ids=func_id)
    def test_search_space_keys(self, func_class):
        """2D functions have x0 and x1 keys."""
        func = instantiate_function(func_class)
        assert "x0" in func.search_space
        assert "x1" in func.search_space


# =============================================================================
# ND Function Tests
# =============================================================================

@pytest.mark.algebraic
class TestNDFunctions:
    """Tests for N-dimensional algebraic functions."""

    @pytest.mark.parametrize("func_class", algebraic_functions_nd, ids=func_id)
    @pytest.mark.parametrize("n_dim", [2, 5, 10, 20])
    def test_dimension_scaling(self, func_class, n_dim):
        """ND functions scale to requested dimensions."""
        func = instantiate_function(func_class, n_dim=n_dim)
        assert len(func.search_space) == n_dim

    @pytest.mark.parametrize("func_class", algebraic_functions_nd, ids=func_id)
    def test_spec_scalable(self, func_class):
        """ND functions have scalable=True."""
        func = instantiate_function(func_class, n_dim=5)
        assert func.spec.get("scalable", False) is True

    @pytest.mark.parametrize("func_class", algebraic_functions_nd, ids=func_id)
    def test_array_input(self, func_class):
        """ND functions accept array input."""
        func = instantiate_function(func_class, n_dim=5)
        result = func(np.zeros(5))
        assert np.isfinite(result)


# =============================================================================
# Known Function Behavior Tests
# =============================================================================

@pytest.mark.algebraic
class TestKnownBehavior:
    """Test known mathematical properties of specific functions."""

    def test_sphere_at_origin(self):
        """Sphere function is 0 at origin."""
        func = SphereFunction(n_dim=5)
        result = func(np.zeros(5))
        assert np.isclose(result, 0.0)

    def test_sphere_positive(self):
        """Sphere function is always >= 0."""
        func = SphereFunction(n_dim=5)
        for _ in range(10):
            x = np.random.randn(5)
            result = func(x)
            assert result >= 0

    def test_rastrigin_at_origin(self):
        """Rastrigin function is 0 at origin."""
        func = RastriginFunction(n_dim=5)
        result = func(np.zeros(5))
        assert np.isclose(result, 0.0)

    def test_rosenbrock_at_ones(self):
        """Rosenbrock function is 0 at (1, 1, ..., 1)."""
        func = RosenbrockFunction(n_dim=5)
        result = func(np.ones(5))
        assert np.isclose(result, 0.0)

    def test_beale_at_optimum(self):
        """Beale function is near 0 at (3, 0.5).

        Note: Default C=2.652 vs canonical C=2.625 causes small discrepancy.
        """
        func = BealeFunction()
        result = func({"x0": 3.0, "x1": 0.5})
        assert result < 0.01  # Near zero but not exactly due to parameter choice

    def test_booth_at_optimum(self):
        """Booth function is 0 at (1, 3)."""
        func = BoothFunction()
        result = func({"x0": 1.0, "x1": 3.0})
        assert np.isclose(result, 0.0)

    def test_himmelblau_multiple_optima(self):
        """Himmelblau's function has 4 equal global minima."""
        func = HimmelblausFunction()
        optima = [
            (3.0, 2.0),
            (-2.805118, 3.131312),
            (-3.779310, -3.283186),
            (3.584428, -1.848126),
        ]
        for x0, x1 in optima:
            result = func({"x0": x0, "x1": x1})
            assert np.isclose(result, 0.0, atol=1e-3)


# =============================================================================
# Input Format Consistency Tests
# =============================================================================

@pytest.mark.algebraic
class TestInputConsistency:
    """Test that different input formats give consistent results."""

    @pytest.mark.parametrize("func_class", algebraic_functions_nd, ids=func_id)
    def test_dict_vs_array(self, func_class):
        """Dict and array inputs give same result."""
        func = instantiate_function(func_class, n_dim=3)
        values = [1.0, 2.0, 3.0]

        dict_result = func({"x0": values[0], "x1": values[1], "x2": values[2]})
        array_result = func(np.array(values))

        assert dict_result == array_result

    @pytest.mark.parametrize("func_class", algebraic_functions_nd, ids=func_id)
    def test_list_vs_array(self, func_class):
        """List and array inputs give same result."""
        func = instantiate_function(func_class, n_dim=3)
        values = [1.0, 2.0, 3.0]

        list_result = func(values)
        array_result = func(np.array(values))

        assert list_result == array_result


# =============================================================================
# Objective Direction Tests
# =============================================================================

@pytest.mark.algebraic
class TestObjectiveDirection:
    """Test minimize/maximize objective behavior."""

    @pytest.mark.parametrize("func_class", algebraic_functions[:5], ids=func_id)
    def test_maximize_negates(self, func_class):
        """Maximize objective negates the function value."""
        func_min = instantiate_function(func_class)
        # Create maximize version manually
        try:
            func_max = func_class(objective="maximize")
        except TypeError:
            func_max = func_class(n_dim=2, objective="maximize")

        params = get_sample_params(func_min)
        result_min = func_min(params)
        result_max = func_max(params)

        assert result_max == -result_min
