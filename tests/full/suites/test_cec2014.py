# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CEC 2014 benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.benchmark.cec.cec2014 import (
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
    # Hybrid
    HybridFunction1,
    HybridFunction2,
    HybridFunction3,
    HybridFunction4,
    HybridFunction5,
    HybridFunction6,
    RotatedBentCigar,
    RotatedDiscus,
    # Unimodal
    RotatedHighConditionedElliptic,
    ShiftedRastrigin,
    ShiftedRotatedAckley,
    ShiftedRotatedExpandedGriewankRosenbrock,
    ShiftedRotatedExpandedScafferF6,
    ShiftedRotatedGriewank,
    ShiftedRotatedHappyCat,
    ShiftedRotatedHGBat,
    ShiftedRotatedKatsuura,
    ShiftedRotatedRastrigin,
    # Multimodal
    ShiftedRotatedRosenbrock,
    ShiftedRotatedSchwefel,
    ShiftedRotatedWeierstrass,
    ShiftedSchwefel,
)

# All CEC 2014 function classes
CEC2014_FUNCTIONS = [
    # Unimodal (F1-F3)
    RotatedHighConditionedElliptic,
    RotatedBentCigar,
    RotatedDiscus,
    # Multimodal (F4-F16)
    ShiftedRotatedRosenbrock,
    ShiftedRotatedAckley,
    ShiftedRotatedWeierstrass,
    ShiftedRotatedGriewank,
    ShiftedRastrigin,
    ShiftedRotatedRastrigin,
    ShiftedSchwefel,
    ShiftedRotatedSchwefel,
    ShiftedRotatedKatsuura,
    ShiftedRotatedHappyCat,
    ShiftedRotatedHGBat,
    ShiftedRotatedExpandedGriewankRosenbrock,
    ShiftedRotatedExpandedScafferF6,
    # Hybrid (F17-F22)
    HybridFunction1,
    HybridFunction2,
    HybridFunction3,
    HybridFunction4,
    HybridFunction5,
    HybridFunction6,
    # Composition (F23-F30)
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
]


class TestCEC2014GlobalOptimum:
    """Test that f(x_global) = f_global for all functions."""

    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS[:16])
    def test_global_optimum_unimodal_multimodal(self, func_class):
        """Test global optimum for unimodal and multimodal functions (F1-F16)."""
        func = func_class(n_dim=10)
        result = func(func.x_global)
        assert np.isclose(
            result, func.f_global, rtol=1e-6
        ), f"{func.name}: f(x_global)={result}, expected {func.f_global}"


class TestCEC2014FunctionProperties:
    """Test function properties and specs."""

    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS)
    def test_has_func_id(self, func_class):
        """Each function must have a func_id."""
        func = func_class(n_dim=10)
        assert func.func_id is not None
        assert 1 <= func.func_id <= 30

    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS)
    def test_f_global_matches_func_id(self, func_class):
        """f_global should be func_id * 100."""
        func = func_class(n_dim=10)
        assert func.f_global == func.func_id * 100

    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS)
    def test_has_spec(self, func_class):
        """Each function must have specs defined."""
        func = func_class(n_dim=10)
        spec = func.spec
        assert "continuous" in spec
        assert "scalable" in spec
        assert spec["scalable"] is True  # All CEC 2014 functions are scalable

    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS[:3])
    def test_unimodal_spec(self, func_class):
        """Unimodal functions (F1-F3) should have unimodal=True."""
        func = func_class(n_dim=10)
        assert func.spec["unimodal"] is True

    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS[3:])
    def test_multimodal_spec(self, func_class):
        """Multimodal functions (F4-F30) should have unimodal=False."""
        func = func_class(n_dim=10)
        assert func.spec["unimodal"] is False


class TestCEC2014Dimensions:
    """Test dimension handling."""

    @pytest.mark.parametrize("dim", [10, 20, 30, 50, 100])
    def test_supported_dimensions(self, dim):
        """Functions should work with all supported dimensions."""
        func = RotatedHighConditionedElliptic(n_dim=dim)
        assert func.n_dim == dim
        result = func(np.zeros(dim))
        assert np.isfinite(result)

    def test_unsupported_dimension_raises(self):
        """Unsupported dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="n_dim must be one of"):
            RotatedHighConditionedElliptic(n_dim=15)


class TestCEC2014InputFormats:
    """Test different input formats."""

    def test_array_input(self):
        """Function should accept numpy array input."""
        func = RotatedHighConditionedElliptic(n_dim=10)
        result = func(np.zeros(10))
        assert np.isfinite(result)

    def test_list_input(self):
        """Function should accept list input."""
        func = RotatedHighConditionedElliptic(n_dim=10)
        result = func([0.0] * 10)
        assert np.isfinite(result)

    def test_dict_input(self):
        """Function should accept dict input."""
        func = RotatedHighConditionedElliptic(n_dim=10)
        params = {f"x{i}": 0.0 for i in range(10)}
        result = func(params)
        assert np.isfinite(result)


class TestCEC2014DataIntegrity:
    """Test data file integrity."""

    def test_rotation_matrix_shape(self):
        """Rotation matrices should have correct shape."""
        func = RotatedHighConditionedElliptic(n_dim=10)
        M = func._get_rotation_matrix()
        assert M.shape == (10, 10)

    def test_shift_vector_in_bounds(self):
        """Shift vectors should be within [-80, 80] (inside search bounds)."""
        func = RotatedHighConditionedElliptic(n_dim=10)
        shift = func._get_shift_vector()
        assert np.all(np.abs(shift) <= 80)

    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS)
    def test_data_loaded_correctly(self, func_class):
        """All functions should load their data without error."""
        func = func_class(n_dim=10)
        # Trigger data loading by evaluating
        result = func(np.zeros(10))
        assert np.isfinite(result)


class TestCEC2014SearchSpace:
    """Test search space properties."""

    def test_default_bounds(self):
        """Default bounds should be [-100, 100]."""
        func = RotatedHighConditionedElliptic(n_dim=10)
        assert func.spec.default_bounds == (-100.0, 100.0)

    def test_search_space(self):
        """Search space should have correct dimensions."""
        func = RotatedHighConditionedElliptic(n_dim=10)
        space = func.search_space
        assert len(space) == 10
        for i in range(10):
            assert f"x{i}" in space


class TestCEC2014Objective:
    """Test objective parameter."""

    def test_minimize_objective(self):
        """Minimize objective should return positive-biased values."""
        func = RotatedHighConditionedElliptic(n_dim=10, objective="minimize")
        result = func(func.x_global)
        assert result == func.f_global  # 100.0

    def test_maximize_objective(self):
        """Maximize objective should negate values."""
        func = RotatedHighConditionedElliptic(n_dim=10, objective="maximize")
        result = func(func.x_global)
        assert result == -func.f_global  # -100.0


COMPOSITION_FUNCTIONS = [
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
]

HYBRID_FUNCTIONS = [
    HybridFunction1,
    HybridFunction2,
    HybridFunction3,
    HybridFunction4,
    HybridFunction5,
    HybridFunction6,
]


class TestCEC2014CompositionEvaluation:
    """Test evaluation of composition functions (F23-F30)."""

    @pytest.mark.parametrize("func_class", COMPOSITION_FUNCTIONS)
    def test_evaluate_at_random_points(self, func_class):
        """Composition functions return finite values at random points."""
        rng = np.random.default_rng(42)
        func = func_class(n_dim=10)
        for _ in range(3):
            x = rng.uniform(-100, 100, size=10)
            result = func(x)
            assert np.isfinite(result), f"{func_class.__name__} returned non-finite"

    @pytest.mark.parametrize("func_class", COMPOSITION_FUNCTIONS)
    def test_evaluate_dict_input(self, func_class):
        """Composition functions accept dict input."""
        func = func_class(n_dim=10)
        params = {f"x{i}": 1.0 for i in range(10)}
        result = func(params)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", COMPOSITION_FUNCTIONS)
    def test_deterministic(self, func_class):
        """Same input produces same output for composition functions."""
        func = func_class(n_dim=10)
        x = np.ones(10) * 10.0
        r1 = func(x)
        r2 = func(x)
        assert r1 == r2


class TestCEC2014HybridEvaluation:
    """Test evaluation of hybrid functions (F17-F22)."""

    @pytest.mark.parametrize("func_class", HYBRID_FUNCTIONS)
    def test_evaluate_at_random_points(self, func_class):
        """Hybrid functions return finite values at random points."""
        rng = np.random.default_rng(42)
        func = func_class(n_dim=10)
        for _ in range(3):
            x = rng.uniform(-100, 100, size=10)
            result = func(x)
            assert np.isfinite(result), f"{func_class.__name__} returned non-finite"

    @pytest.mark.parametrize("func_class", HYBRID_FUNCTIONS)
    def test_evaluate_dict_input(self, func_class):
        """Hybrid functions accept dict input."""
        func = func_class(n_dim=10)
        params = {f"x{i}": 1.0 for i in range(10)}
        result = func(params)
        assert np.isfinite(result)


class TestCEC2014BatchEvaluation:
    """Test batch evaluation for composition and hybrid functions."""

    @pytest.mark.parametrize("func_class", COMPOSITION_FUNCTIONS)
    def test_batch_composition_matches_scalar(self, func_class):
        """Batch composition matches scalar evaluation."""
        rng = np.random.default_rng(42)
        func = func_class(n_dim=10)
        X = rng.uniform(-50, 50, size=(3, 10))

        batch_results = func.batch(X)
        scalar_results = np.array([func(X[i]) for i in range(3)])

        np.testing.assert_allclose(batch_results, scalar_results, rtol=1e-5)

    @pytest.mark.parametrize("func_class", HYBRID_FUNCTIONS)
    def test_batch_hybrid_matches_scalar(self, func_class):
        """Batch hybrid matches scalar evaluation."""
        rng = np.random.default_rng(42)
        func = func_class(n_dim=10)
        X = rng.uniform(-50, 50, size=(3, 10))

        batch_results = func.batch(X)
        scalar_results = np.array([func(X[i]) for i in range(3)])

        np.testing.assert_allclose(batch_results, scalar_results, rtol=1e-5)

    @pytest.mark.parametrize("func_class", COMPOSITION_FUNCTIONS[:2] + HYBRID_FUNCTIONS[:2])
    def test_batch_shape(self, func_class):
        """Batch returns correct shape."""
        func = func_class(n_dim=10)
        X = np.zeros((5, 10))
        result = func.batch(X)
        assert result.shape == (5,)
