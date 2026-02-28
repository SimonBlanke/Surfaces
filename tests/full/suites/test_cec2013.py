# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CEC 2013 benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.benchmark.cec.cec2013 import (
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
    DifferentPowers,
    LunacekBiRastrigin,
    Rastrigin,
    RotatedAckley,
    RotatedBentCigar,
    RotatedDiscus,
    RotatedExpandedGriewankRosenbrock,
    RotatedExpandedScafferF6,
    RotatedGriewank,
    RotatedHighConditionedElliptic,
    RotatedKatsuura,
    RotatedLunacekBiRastrigin,
    RotatedRastrigin,
    RotatedRosenbrock,
    RotatedSchafferF7,
    RotatedSchwefel,
    RotatedWeierstrass,
    Schwefel,
    Sphere,
    StepRastrigin,
    cec2013_functions,
)

# ---- Function categories ----
UNIMODAL = [
    Sphere,
    RotatedHighConditionedElliptic,
    RotatedBentCigar,
    RotatedDiscus,
    DifferentPowers,
]
MULTIMODAL = [
    RotatedRosenbrock,
    RotatedSchafferF7,
    RotatedAckley,
    RotatedWeierstrass,
    RotatedGriewank,
    Rastrigin,
    RotatedRastrigin,
    StepRastrigin,
    Schwefel,
    RotatedSchwefel,
    RotatedKatsuura,
    LunacekBiRastrigin,
    RotatedLunacekBiRastrigin,
    RotatedExpandedGriewankRosenbrock,
    RotatedExpandedScafferF6,
]
COMPOSITION = [
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
]

ALL_FUNCTIONS = UNIMODAL + MULTIMODAL + COMPOSITION


class TestCEC2013Evaluation:
    """Test evaluation of all CEC 2013 functions."""

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS)
    def test_evaluate_at_zeros(self, func_class):
        """All functions return finite values at the origin."""
        func = func_class(n_dim=10)
        result = func(np.zeros(10))
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS)
    def test_evaluate_at_random(self, func_class):
        """All functions return finite values at random points."""
        rng = np.random.default_rng(42)
        func = func_class(n_dim=10)
        x = rng.uniform(-100, 100, size=10)
        result = func(x)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS)
    def test_evaluate_dict_input(self, func_class):
        """All functions accept dict input."""
        func = func_class(n_dim=10)
        params = {f"x{i}": 0.0 for i in range(10)}
        result = func(params)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS)
    def test_evaluate_list_input(self, func_class):
        """All functions accept list input."""
        func = func_class(n_dim=10)
        result = func([0.0] * 10)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS)
    def test_multiple_evaluations_deterministic(self, func_class):
        """Same input produces same output."""
        func = func_class(n_dim=10)
        x = np.ones(10) * 5.0
        r1 = func(x)
        r2 = func(x)
        assert r1 == r2


class TestCEC2013BatchEvaluation:
    """Test batch evaluation for CEC 2013 functions."""

    @pytest.mark.parametrize("func_class", UNIMODAL + MULTIMODAL)
    def test_batch_matches_scalar(self, func_class):
        """Batch evaluation matches sequential scalar evaluation."""
        rng = np.random.default_rng(42)
        func = func_class(n_dim=10)
        X = rng.uniform(-50, 50, size=(5, 10))

        batch_results = func.batch(X)
        scalar_results = np.array([func(X[i]) for i in range(5)])

        np.testing.assert_allclose(batch_results, scalar_results, rtol=1e-5)

    @pytest.mark.parametrize("func_class", COMPOSITION)
    def test_batch_composition_matches_scalar(self, func_class):
        """Batch composition functions match scalar evaluation."""
        rng = np.random.default_rng(42)
        func = func_class(n_dim=10)
        X = rng.uniform(-50, 50, size=(3, 10))

        batch_results = func.batch(X)
        scalar_results = np.array([func(X[i]) for i in range(3)])

        np.testing.assert_allclose(batch_results, scalar_results, rtol=1e-5)

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS[:5])
    def test_batch_shape(self, func_class):
        """Batch returns correct shape."""
        func = func_class(n_dim=10)
        X = np.zeros((8, 10))
        result = func.batch(X)
        assert result.shape == (8,)


class TestCEC2013Properties:
    """Test function properties and specs."""

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS)
    def test_has_func_id(self, func_class):
        """Each function has a valid func_id."""
        func = func_class(n_dim=10)
        assert func.func_id is not None
        assert 1 <= func.func_id <= 28

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS)
    def test_f_global_matches_convention(self, func_class):
        """f_global follows the -1400 + (func_id - 1) * 100 convention."""
        func = func_class(n_dim=10)
        expected = -1400 + (func.func_id - 1) * 100
        assert func.f_global == expected

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS)
    def test_has_search_space(self, func_class):
        """Each function has a search space."""
        func = func_class(n_dim=10)
        space = func.search_space
        assert len(space) == 10

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS)
    def test_has_spec(self, func_class):
        """Each function has a spec dict."""
        func = func_class(n_dim=10)
        spec = func.spec
        assert isinstance(spec.as_dict(), dict)

    def test_function_count(self):
        """cec2013_functions contains all 28 functions."""
        assert len(cec2013_functions) == 28


class TestCEC2013Dimensions:
    """Test dimension handling."""

    @pytest.mark.parametrize("dim", [10, 20, 30, 50])
    def test_supported_dimensions(self, dim):
        """Functions work with supported dimensions."""
        func = Sphere(n_dim=dim)
        assert func.n_dim == dim
        result = func(np.zeros(dim))
        assert np.isfinite(result)

    def test_unsupported_dimension_raises(self):
        """Unsupported dimensions raise ValueError."""
        with pytest.raises(ValueError, match="n_dim must be one of"):
            Sphere(n_dim=15)


class TestCEC2013GlobalOptimum:
    """Test global optimum evaluation for unimodal functions."""

    @pytest.mark.parametrize("func_class", UNIMODAL)
    def test_global_optimum(self, func_class):
        """f(x_global) = f_global for unimodal functions."""
        func = func_class(n_dim=10)
        result = func(func.x_global)
        assert np.isclose(result, func.f_global, rtol=1e-6)
