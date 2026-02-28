# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CEC 2017 benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.benchmark.cec.cec2017 import (
    ShiftedRotatedBentCigar,
    cec2017_functions,
)

ALL_FUNCTIONS = cec2017_functions


class TestCEC2017Evaluation:
    """Test evaluation of all CEC 2017 simple functions."""

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
    def test_evaluate_at_multiple_random_points(self, func_class):
        """Functions return finite values at multiple random points."""
        rng = np.random.default_rng(123)
        func = func_class(n_dim=10)
        for _ in range(5):
            x = rng.uniform(-100, 100, size=10)
            result = func(x)
            assert np.isfinite(result)


class TestCEC2017BatchEvaluation:
    """Test batch evaluation for CEC 2017 functions."""

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS)
    def test_batch_matches_scalar(self, func_class):
        """Batch evaluation matches sequential scalar evaluation."""
        rng = np.random.default_rng(42)
        func = func_class(n_dim=10)
        X = rng.uniform(-50, 50, size=(5, 10))

        batch_results = func.batch(X)
        scalar_results = np.array([func(X[i]) for i in range(5)])

        np.testing.assert_allclose(batch_results, scalar_results, rtol=1e-6)

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS[:3])
    def test_batch_shape(self, func_class):
        """Batch returns correct shape."""
        func = func_class(n_dim=10)
        X = np.zeros((8, 10))
        result = func.batch(X)
        assert result.shape == (8,)


class TestCEC2017Properties:
    """Test function properties and specs."""

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS)
    def test_has_func_id(self, func_class):
        """Each function has a valid func_id."""
        func = func_class(n_dim=10)
        assert func.func_id is not None
        assert 1 <= func.func_id <= 10

    @pytest.mark.parametrize("func_class", ALL_FUNCTIONS)
    def test_f_global_matches_convention(self, func_class):
        """f_global follows the func_id * 100 convention."""
        func = func_class(n_dim=10)
        assert func.f_global == func.func_id * 100

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
        assert "scalable" in spec

    def test_function_count(self):
        """cec2017_functions contains all 10 simple functions."""
        assert len(cec2017_functions) == 10


class TestCEC2017Dimensions:
    """Test dimension handling."""

    @pytest.mark.parametrize("dim", [10, 20, 30, 50])
    def test_supported_dimensions(self, dim):
        """Functions work with supported dimensions."""
        func = ShiftedRotatedBentCigar(n_dim=dim)
        assert func.n_dim == dim
        result = func(np.zeros(dim))
        assert np.isfinite(result)

    def test_unsupported_dimension_raises(self):
        """Unsupported dimensions raise ValueError."""
        with pytest.raises(ValueError, match="n_dim must be one of"):
            ShiftedRotatedBentCigar(n_dim=15)


class TestCEC2017Objective:
    """Test objective parameter."""

    def test_minimize_objective(self):
        """Minimize objective returns positive-biased values."""
        func = ShiftedRotatedBentCigar(n_dim=10, objective="minimize")
        result = func(func.x_global)
        assert result == func.f_global

    def test_maximize_objective(self):
        """Maximize objective negates values."""
        func = ShiftedRotatedBentCigar(n_dim=10, objective="maximize")
        result = func(func.x_global)
        assert result == -func.f_global
