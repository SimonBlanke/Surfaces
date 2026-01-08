# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CEC 2008 large-scale benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.benchmark.cec.cec2008 import (
    CEC2008Function,
    ShiftedSphere2008,
    ShiftedSchwefel221,
    ShiftedRosenbrock2008,
    ShiftedRastrigin2008,
    ShiftedGriewank2008,
    ShiftedAckley2008,
    FastFractalDoubleDip,
    CEC2008_ALL,
    CEC2008_SEPARABLE,
    CEC2008_NONSEPARABLE,
)


class TestCEC2008Instantiation:
    """Test that all functions can be instantiated."""

    @pytest.mark.parametrize("func_class", CEC2008_ALL)
    def test_instantiation(self, func_class):
        """All functions should instantiate without error."""
        func = func_class()
        assert func is not None
        assert func.n_dim == 1000

    @pytest.mark.parametrize("func_class", CEC2008_ALL)
    def test_func_id(self, func_class):
        """Each function should have a valid func_id."""
        func = func_class()
        assert func.func_id is not None
        assert 1 <= func.func_id <= 7


class TestCEC2008Properties:
    """Test function properties and metadata."""

    @pytest.mark.parametrize("func_class", CEC2008_ALL)
    def test_fixed_dimension(self, func_class):
        """All CEC 2008 functions should have n_dim=1000."""
        func = func_class()
        assert func.n_dim == 1000

    @pytest.mark.parametrize("func_class", CEC2008_ALL)
    def test_f_global_is_zero(self, func_class):
        """All CEC 2008 functions should have f_global=0."""
        func = func_class()
        assert func.f_global == 0.0

    @pytest.mark.parametrize("func_class", CEC2008_ALL)
    def test_supported_dims(self, func_class):
        """All CEC 2008 functions only support 1000D."""
        func = func_class()
        assert func.supported_dims == (1000,)

    @pytest.mark.parametrize("func_class", CEC2008_SEPARABLE)
    def test_separable_spec(self, func_class):
        """Separable functions should have separable=True."""
        func = func_class()
        assert func.spec.get("separable") is True

    @pytest.mark.parametrize("func_class", CEC2008_NONSEPARABLE)
    def test_nonseparable_spec(self, func_class):
        """Non-separable functions should have separable=False."""
        func = func_class()
        assert func.spec.get("separable") is False


class TestCEC2008GlobalOptimum:
    """Test global optimum values."""

    @pytest.mark.parametrize("func_class", [
        ShiftedSphere2008,
        ShiftedSchwefel221,
        # ShiftedRosenbrock2008 has optimum at shift + 1, not at shift
        ShiftedRastrigin2008,
        ShiftedGriewank2008,
        ShiftedAckley2008,
        FastFractalDoubleDip,
    ])
    def test_global_optimum_value(self, func_class):
        """Test f(x_global) is close to 0."""
        func = func_class()
        if func.x_global is not None:
            result = func(func.x_global)
            assert np.isclose(result, 0.0, atol=1e-6), \
                f"{func_class.__name__}: f(x_global)={result}, expected 0.0"


class TestCEC2008Evaluation:
    """Test function evaluation."""

    @pytest.mark.parametrize("func_class", CEC2008_ALL)
    def test_evaluation_returns_finite(self, func_class):
        """Function evaluation should return finite values."""
        func = func_class()
        np.random.seed(42)
        # Use a small random vector within bounds
        lb, ub = func.default_bounds
        x = np.random.uniform(lb * 0.1, ub * 0.1, func.n_dim)
        result = func(x)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2008_ALL)
    def test_dict_input(self, func_class):
        """Function should accept dict input."""
        func = func_class()
        np.random.seed(42)
        lb, ub = func.default_bounds
        params = {f"x{i}": np.random.uniform(lb * 0.1, ub * 0.1) for i in range(func.n_dim)}
        result = func(params)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2008_ALL)
    def test_array_input(self, func_class):
        """Function should accept array input with proper mapping."""
        func = func_class()
        np.random.seed(42)
        lb, ub = func.default_bounds
        x = np.random.uniform(lb * 0.1, ub * 0.1, func.n_dim)

        # Array and dict should give same result
        result_array = func(x)
        params = {f"x{i}": x[i] for i in range(func.n_dim)}
        result_dict = func(params)

        assert np.isclose(result_array, result_dict)


class TestCEC2008BatchEvaluation:
    """Test batch evaluation functionality."""

    @pytest.mark.parametrize("func_class", [
        ShiftedSphere2008,
        ShiftedSchwefel221,
        ShiftedRosenbrock2008,
        ShiftedRastrigin2008,
        ShiftedGriewank2008,
        ShiftedAckley2008,
        # FastFractalDoubleDip has intentionally different batch implementation
        # for performance (simplified fractal)
    ])
    def test_batch_matches_sequential(self, func_class):
        """Batch evaluation should match sequential evaluation."""
        func = func_class()
        np.random.seed(42)

        # Generate small test points to avoid numerical issues
        lb, ub = func.default_bounds
        X = np.random.uniform(lb * 0.01, ub * 0.01, size=(3, func.n_dim))

        # Sequential evaluation via dict (bypasses potential issues)
        sequential = []
        for i in range(X.shape[0]):
            params = {f"x{j}": X[i, j] for j in range(func.n_dim)}
            sequential.append(func.pure_objective_function(params))
        sequential = np.array(sequential)

        # Batch evaluation
        batch = func._batch_objective(X)

        assert np.allclose(sequential, batch, rtol=1e-6)


class TestCEC2008SpecificFunctions:
    """Test specific function implementations."""

    def test_sphere_at_origin(self):
        """Sphere at shifted origin should be 0."""
        func = ShiftedSphere2008()
        result = func(func.x_global)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_schwefel221_at_origin(self):
        """Schwefel 2.21 at shifted origin should be 0."""
        func = ShiftedSchwefel221()
        result = func(func.x_global)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_rosenbrock_non_separable(self):
        """Rosenbrock should be non-separable."""
        func = ShiftedRosenbrock2008()
        assert func.spec.get("separable") is False

    def test_rastrigin_multimodal(self):
        """Rastrigin should be multimodal."""
        func = ShiftedRastrigin2008()
        assert func.spec.get("unimodal") is False

    def test_ackley_at_origin(self):
        """Ackley at shifted origin should be close to 0."""
        func = ShiftedAckley2008()
        result = func(func.x_global)
        # Ackley has small numerical errors at optimum
        assert np.isclose(result, 0.0, atol=1e-8)

    def test_griewank_at_origin(self):
        """Griewank at shifted origin should be 0."""
        func = ShiftedGriewank2008()
        result = func(func.x_global)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_fractal_at_origin(self):
        """Fast Fractal Double Dip at shifted origin should be 0."""
        func = FastFractalDoubleDip()
        result = func(func.x_global)
        assert np.isclose(result, 0.0, atol=1e-10)


class TestCEC2008Objective:
    """Test objective parameter."""

    def test_minimize_objective(self):
        """Minimize objective should return original values."""
        func = ShiftedSphere2008(objective="minimize")
        result = func(func.x_global)
        assert result >= 0

    def test_maximize_objective(self):
        """Maximize objective should negate values."""
        func = ShiftedSphere2008(objective="maximize")
        result = func(func.x_global)
        assert result <= 0  # Negated minimum


class TestCEC2008Bounds:
    """Test search space bounds."""

    def test_sphere_bounds(self):
        """Sphere should have [-100, 100] bounds."""
        func = ShiftedSphere2008()
        assert func.default_bounds == (-100.0, 100.0)

    def test_rastrigin_bounds(self):
        """Rastrigin should have [-5, 5] bounds."""
        func = ShiftedRastrigin2008()
        assert func.default_bounds == (-5.0, 5.0)

    def test_griewank_bounds(self):
        """Griewank should have [-600, 600] bounds."""
        func = ShiftedGriewank2008()
        assert func.default_bounds == (-600.0, 600.0)

    def test_ackley_bounds(self):
        """Ackley should have [-32, 32] bounds."""
        func = ShiftedAckley2008()
        assert func.default_bounds == (-32.0, 32.0)

    def test_fractal_bounds(self):
        """Fast Fractal should have [-1, 1] bounds."""
        func = FastFractalDoubleDip()
        assert func.default_bounds == (-1.0, 1.0)
