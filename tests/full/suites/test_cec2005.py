# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CEC 2005 benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.benchmark.cec.cec2005 import (
    # Unimodal (F1-F5)
    ShiftedSphere,
    ShiftedSchwefel12,
    ShiftedRotatedElliptic,
    ShiftedSchwefel12Noise,
    SchwefelProblem26,
    # Multimodal (F6-F14)
    ShiftedRosenbrock,
    ShiftedRotatedGriewank,
    ShiftedRotatedAckley,
    ShiftedRastrigin,
    ShiftedRotatedRastrigin,
    ShiftedRotatedWeierstrass,
    SchwefelProblem213,
    ExpandedGriewankRosenbrock,
    ShiftedRotatedExpandedScaffer,
    # Composition (F15-F25)
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
    CompositionFunction9,
    CompositionFunction10,
    CompositionFunction11,
)

# All CEC 2005 function classes
CEC2005_UNIMODAL = [
    ShiftedSphere,  # F1
    ShiftedSchwefel12,  # F2
    ShiftedRotatedElliptic,  # F3
    ShiftedSchwefel12Noise,  # F4
    SchwefelProblem26,  # F5
]

CEC2005_MULTIMODAL = [
    ShiftedRosenbrock,  # F6
    ShiftedRotatedGriewank,  # F7
    ShiftedRotatedAckley,  # F8
    ShiftedRastrigin,  # F9
    ShiftedRotatedRastrigin,  # F10
    ShiftedRotatedWeierstrass,  # F11
    SchwefelProblem213,  # F12
    ExpandedGriewankRosenbrock,  # F13
    ShiftedRotatedExpandedScaffer,  # F14
]

CEC2005_COMPOSITION = [
    CompositionFunction1,  # F15
    CompositionFunction2,  # F16
    CompositionFunction3,  # F17
    CompositionFunction4,  # F18
    CompositionFunction5,  # F19
    CompositionFunction6,  # F20
    CompositionFunction7,  # F21
    CompositionFunction8,  # F22
    CompositionFunction9,  # F23
    CompositionFunction10,  # F24
    CompositionFunction11,  # F25
]

CEC2005_ALL = CEC2005_UNIMODAL + CEC2005_MULTIMODAL + CEC2005_COMPOSITION

# Rotated functions that require specific dimensions
CEC2005_ROTATED = [
    ShiftedRotatedElliptic,
    ShiftedRotatedGriewank,
    ShiftedRotatedAckley,
    ShiftedRotatedRastrigin,
    ShiftedRotatedWeierstrass,
    ExpandedGriewankRosenbrock,
    ShiftedRotatedExpandedScaffer,
] + CEC2005_COMPOSITION

# Non-rotated functions that support arbitrary dimensions
CEC2005_NON_ROTATED = [
    ShiftedSphere,
    ShiftedSchwefel12,
    ShiftedSchwefel12Noise,
    SchwefelProblem26,
    ShiftedRosenbrock,
    ShiftedRastrigin,
    SchwefelProblem213,
]

# Functions without noise (deterministic)
CEC2005_DETERMINISTIC = [f for f in CEC2005_ALL if f not in [
    ShiftedSchwefel12Noise,  # F4
    CompositionFunction3,  # F17
]]


class TestCEC2005Instantiation:
    """Test that all functions can be instantiated."""

    @pytest.mark.parametrize("func_class", CEC2005_ALL)
    def test_instantiation(self, func_class):
        """All functions should instantiate without error."""
        func = func_class(n_dim=10)
        assert func is not None
        assert func.n_dim == 10


class TestCEC2005GlobalOptimum:
    """Test global optimum values."""

    @pytest.mark.parametrize("func_class", CEC2005_DETERMINISTIC[:14])
    def test_global_optimum_basic(self, func_class):
        """Test f(x_global) is close to f_global for basic functions."""
        func = func_class(n_dim=10)
        if func.x_global is not None:
            result = func(func.x_global)
            # Allow some tolerance for numerical errors
            assert np.isclose(result, func.f_global, rtol=1e-4, atol=1e-4), \
                f"{func.name}: f(x_global)={result}, expected {func.f_global}"


class TestCEC2005FunctionProperties:
    """Test function properties and specs."""

    @pytest.mark.parametrize("func_class", CEC2005_ALL)
    def test_has_func_id(self, func_class):
        """Each function must have a func_id."""
        func = func_class(n_dim=10)
        assert func.func_id is not None
        assert 1 <= func.func_id <= 25

    @pytest.mark.parametrize("func_class", CEC2005_ALL)
    def test_f_global_is_bias(self, func_class):
        """f_global should match the CEC2005 bias values."""
        func = func_class(n_dim=10)
        # CEC2005 uses fixed bias values per function
        expected_biases = {
            1: -450, 2: -450, 3: -450, 4: -450, 5: -310,
            6: 390, 7: -180, 8: -140, 9: -330, 10: -330,
            11: 90, 12: -460, 13: -130, 14: -300,
            15: 120, 16: 120, 17: 120, 18: 10, 19: 10,
            20: 10, 21: 360, 22: 360, 23: 360, 24: 260, 25: 260
        }
        assert func.f_global == expected_biases[func.func_id]

    @pytest.mark.parametrize("func_class", CEC2005_ALL)
    def test_has_spec(self, func_class):
        """Each function must have specs defined."""
        func = func_class(n_dim=10)
        spec = func.spec
        assert "continuous" in spec
        assert "scalable" in spec

    @pytest.mark.parametrize("func_class", CEC2005_UNIMODAL)
    def test_unimodal_spec(self, func_class):
        """Unimodal functions (F1-F5) should have unimodal=True."""
        func = func_class(n_dim=10)
        assert func.spec["unimodal"] is True

    @pytest.mark.parametrize("func_class", CEC2005_MULTIMODAL + CEC2005_COMPOSITION)
    def test_multimodal_spec(self, func_class):
        """Multimodal functions (F6-F25) should have unimodal=False."""
        func = func_class(n_dim=10)
        assert func.spec["unimodal"] is False


class TestCEC2005Dimensions:
    """Test dimension handling."""

    @pytest.mark.parametrize("dim", [2, 10, 30, 50])
    def test_rotated_supported_dimensions(self, dim):
        """Rotated functions should work with supported dimensions."""
        func = ShiftedRotatedElliptic(n_dim=dim)
        assert func.n_dim == dim
        result = func(np.zeros(dim))
        assert np.isfinite(result)

    def test_rotated_unsupported_dimension_raises(self):
        """Rotated functions should reject unsupported dimensions."""
        with pytest.raises(ValueError, match="n_dim must be one of"):
            ShiftedRotatedElliptic(n_dim=15)

    def test_all_functions_same_dimensions(self):
        """All CEC2005 functions support the same dimensions (2, 10, 30, 50)."""
        # CEC2005 data files are only available for these dimensions
        func = ShiftedSphere(n_dim=10)
        assert func.supported_dims == (2, 10, 30, 50)


class TestCEC2005InputFormats:
    """Test different input formats."""

    def test_array_input(self):
        """Function should accept numpy array input."""
        func = ShiftedSphere(n_dim=10)
        result = func(np.zeros(10))
        assert np.isfinite(result)

    def test_list_input(self):
        """Function should accept list input."""
        func = ShiftedSphere(n_dim=10)
        result = func([0.0] * 10)
        assert np.isfinite(result)

    def test_dict_input(self):
        """Function should accept dict input."""
        func = ShiftedSphere(n_dim=10)
        params = {f"x{i}": 0.0 for i in range(10)}
        result = func(params)
        assert np.isfinite(result)


class TestCEC2005DataIntegrity:
    """Test data file integrity."""

    @pytest.mark.parametrize("func_class", CEC2005_ROTATED[:7])
    def test_rotation_matrix_shape(self, func_class):
        """Rotation matrices should have correct shape."""
        func = func_class(n_dim=10)
        M = func._get_rotation_matrix()
        assert M.shape == (10, 10)

    @pytest.mark.parametrize("func_class", CEC2005_ALL)
    def test_data_loaded_correctly(self, func_class):
        """All functions should load their data without error."""
        func = func_class(n_dim=10)
        # Trigger data loading by evaluating
        result = func(np.zeros(10))
        assert np.isfinite(result)


class TestCEC2005Bounds:
    """Test search space bounds."""

    def test_basic_bounds(self):
        """F1-F8 should have [-100, 100] bounds by default."""
        func = ShiftedSphere(n_dim=10)
        assert func.default_bounds == (-100.0, 100.0)

    def test_rastrigin_bounds(self):
        """F9, F10 should have [-5, 5] bounds."""
        func = ShiftedRastrigin(n_dim=10)
        assert func.default_bounds == (-5.0, 5.0)

    def test_weierstrass_bounds(self):
        """F11 should have [-0.5, 0.5] bounds."""
        func = ShiftedRotatedWeierstrass(n_dim=10)
        assert func.default_bounds == (-0.5, 0.5)

    def test_composition_bounds(self):
        """F15-F25 should have [-5, 5] bounds."""
        func = CompositionFunction1(n_dim=10)
        assert func.default_bounds == (-5.0, 5.0)


class TestCEC2005BatchEvaluation:
    """Test batch evaluation functionality."""

    @pytest.mark.parametrize("func_class", CEC2005_DETERMINISTIC[:14])
    def test_batch_matches_sequential(self, func_class):
        """Batch evaluation should match sequential evaluation."""
        func = func_class(n_dim=10)

        # Generate random test points
        np.random.seed(42)
        X = np.random.uniform(-1, 1, size=(5, 10))

        # Sequential evaluation
        sequential = np.array([func(x) for x in X])

        # Batch evaluation
        batch = func._batch_objective(X)

        assert np.allclose(sequential, batch, rtol=1e-10)


class TestCEC2005Objective:
    """Test objective parameter."""

    def test_minimize_objective(self):
        """Minimize objective should return original values."""
        func = ShiftedSphere(n_dim=10, objective="minimize")
        result = func(func.x_global)
        assert result == func.f_global

    def test_maximize_objective(self):
        """Maximize objective should negate values."""
        func = ShiftedSphere(n_dim=10, objective="maximize")
        result = func(func.x_global)
        assert result == -func.f_global
