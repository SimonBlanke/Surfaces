# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CEC 2010 large-scale benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.benchmark.cec.cec2010 import (
    CEC2010Function,
    # Separable (F1-F3)
    SeparableElliptic,
    SeparableRastrigin,
    SeparableAckley,
    # Single-group (F4-F7)
    SingleGroupElliptic,
    SingleGroupRastrigin,
    SingleGroupAckley,
    SingleGroupSchwefel,
    # Multi-group (F8-F13)
    MultiGroupElliptic,
    MultiGroupRastrigin,
    MultiGroupAckley,
    MultiGroupSchwefel,
    MultiGroupRosenbrock,
    MultiGroupGriewank,
    # Non-separable (F14-F18)
    OverlapSchwefel,
    OverlapRosenbrock,
    NonSepRastrigin,
    NonSepAckley,
    NonSepGriewank,
    # Composition (F19-F20)
    Composition1,
    Composition2,
    # Collections
    CEC2010_ALL,
    CEC2010_SEPARABLE,
    CEC2010_PARTIAL_SEPARABLE,
    CEC2010_NONSEPARABLE,
    CEC2010_COMPOSITION,
)


class TestCEC2010Instantiation:
    """Test that all functions can be instantiated."""

    @pytest.mark.parametrize("func_class", CEC2010_ALL)
    def test_instantiation(self, func_class):
        """All functions should instantiate without error."""
        func = func_class()
        assert func is not None
        assert func.n_dim == 1000

    @pytest.mark.parametrize("func_class", CEC2010_ALL)
    def test_func_id(self, func_class):
        """Each function should have a valid func_id."""
        func = func_class()
        assert func.func_id is not None
        assert 1 <= func.func_id <= 20


class TestCEC2010Properties:
    """Test function properties and metadata."""

    @pytest.mark.parametrize("func_class", CEC2010_ALL)
    def test_fixed_dimension(self, func_class):
        """All CEC 2010 functions should have n_dim=1000."""
        func = func_class()
        assert func.n_dim == 1000

    @pytest.mark.parametrize("func_class", CEC2010_ALL)
    def test_f_global_is_zero(self, func_class):
        """All CEC 2010 functions should have f_global=0."""
        func = func_class()
        assert func.f_global == 0.0

    @pytest.mark.parametrize("func_class", CEC2010_ALL)
    def test_supported_dims(self, func_class):
        """All CEC 2010 functions only support 1000D."""
        func = func_class()
        assert func.supported_dims == (1000,)

    @pytest.mark.parametrize("func_class", CEC2010_SEPARABLE)
    def test_separable_spec(self, func_class):
        """Separable functions should have separable=True."""
        func = func_class()
        assert func.spec.get("separable") is True

    @pytest.mark.parametrize("func_class", CEC2010_PARTIAL_SEPARABLE + CEC2010_NONSEPARABLE)
    def test_nonseparable_spec(self, func_class):
        """Non-separable functions should have separable=False."""
        func = func_class()
        assert func.spec.get("separable") is False

    def test_group_parameters(self):
        """Check partial separability parameters."""
        func = SeparableElliptic()
        assert func.m == 50  # Group size
        assert func.n_groups == 20  # Number of groups


class TestCEC2010Evaluation:
    """Test function evaluation."""

    @pytest.mark.parametrize("func_class", CEC2010_ALL)
    def test_evaluation_returns_finite(self, func_class):
        """Function evaluation should return finite values."""
        func = func_class()
        np.random.seed(42)
        # Use a small random vector within bounds
        lb, ub = func.default_bounds
        x = np.random.uniform(lb * 0.01, ub * 0.01, func.n_dim)
        result = func(x)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2010_ALL)
    def test_dict_input(self, func_class):
        """Function should accept dict input."""
        func = func_class()
        np.random.seed(42)
        lb, ub = func.default_bounds
        params = {f"x{i}": np.random.uniform(lb * 0.01, ub * 0.01) for i in range(func.n_dim)}
        result = func(params)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2010_ALL)
    def test_array_input(self, func_class):
        """Function should accept array input with proper mapping."""
        func = func_class()
        np.random.seed(42)
        lb, ub = func.default_bounds
        x = np.random.uniform(lb * 0.01, ub * 0.01, func.n_dim)

        # Array and dict should give same result
        result_array = func(x)
        params = {f"x{i}": x[i] for i in range(func.n_dim)}
        result_dict = func(params)

        assert np.isclose(result_array, result_dict)


class TestCEC2010BatchEvaluation:
    """Test batch evaluation functionality."""

    @pytest.mark.parametrize("func_class", CEC2010_SEPARABLE)
    def test_separable_batch_matches_sequential(self, func_class):
        """Batch evaluation should match sequential for separable functions."""
        func = func_class()
        np.random.seed(42)

        # Generate small test points
        lb, ub = func.default_bounds
        X = np.random.uniform(lb * 0.001, ub * 0.001, size=(3, func.n_dim))

        # Sequential evaluation via dict
        sequential = []
        for i in range(X.shape[0]):
            params = {f"x{j}": X[i, j] for j in range(func.n_dim)}
            sequential.append(func.pure_objective_function(params))
        sequential = np.array(sequential)

        # Batch evaluation
        batch = func._batch_objective(X)

        assert np.allclose(sequential, batch, rtol=1e-6)

    @pytest.mark.parametrize("func_class", [
        MultiGroupElliptic,  # Representative from each category
        NonSepRastrigin,
        Composition1,
    ])
    def test_partial_sep_batch_matches_sequential(self, func_class):
        """Batch evaluation should match sequential for partial separable functions."""
        func = func_class()
        np.random.seed(42)

        lb, ub = func.default_bounds
        X = np.random.uniform(lb * 0.001, ub * 0.001, size=(2, func.n_dim))

        sequential = []
        for i in range(X.shape[0]):
            params = {f"x{j}": X[i, j] for j in range(func.n_dim)}
            sequential.append(func.pure_objective_function(params))
        sequential = np.array(sequential)

        batch = func._batch_objective(X)

        assert np.allclose(sequential, batch, rtol=1e-4)


class TestCEC2010FunctionCategories:
    """Test function category collections."""

    def test_all_functions_count(self):
        """CEC2010_ALL should contain exactly 20 functions."""
        assert len(CEC2010_ALL) == 20

    def test_separable_count(self):
        """CEC2010_SEPARABLE should contain exactly 3 functions (F1-F3)."""
        assert len(CEC2010_SEPARABLE) == 3

    def test_partial_separable_count(self):
        """CEC2010_PARTIAL_SEPARABLE should contain 10 functions (F4-F13)."""
        assert len(CEC2010_PARTIAL_SEPARABLE) == 10

    def test_nonseparable_count(self):
        """CEC2010_NONSEPARABLE should contain 5 functions (F14-F18)."""
        assert len(CEC2010_NONSEPARABLE) == 5

    def test_composition_count(self):
        """CEC2010_COMPOSITION should contain 2 functions (F19-F20)."""
        assert len(CEC2010_COMPOSITION) == 2

    def test_categories_sum_to_all(self):
        """All category collections should sum to CEC2010_ALL."""
        all_from_categories = set(
            CEC2010_SEPARABLE +
            CEC2010_PARTIAL_SEPARABLE +
            CEC2010_NONSEPARABLE +
            CEC2010_COMPOSITION
        )
        assert len(all_from_categories) == len(CEC2010_ALL)


class TestCEC2010SpecificFunctions:
    """Test specific function implementations."""

    def test_separable_elliptic_id(self):
        """SeparableElliptic should have func_id=1."""
        func = SeparableElliptic()
        assert func.func_id == 1

    def test_composition2_id(self):
        """Composition2 should have func_id=20."""
        func = Composition2()
        assert func.func_id == 20

    def test_func_ids_sequential(self):
        """Function IDs should be sequential from 1 to 20."""
        func_ids = sorted([cls().func_id for cls in CEC2010_ALL])
        assert func_ids == list(range(1, 21))


class TestCEC2010Objective:
    """Test objective parameter."""

    def test_minimize_objective(self):
        """Minimize objective should return original values."""
        func = SeparableElliptic(objective="minimize")
        np.random.seed(42)
        lb, ub = func.default_bounds
        x = np.random.uniform(lb * 0.01, ub * 0.01, func.n_dim)
        result = func(x)
        assert result >= 0

    def test_maximize_objective(self):
        """Maximize objective should negate values."""
        func = SeparableElliptic(objective="maximize")
        np.random.seed(42)
        lb, ub = func.default_bounds
        x = np.random.uniform(lb * 0.01, ub * 0.01, func.n_dim)
        result = func(x)
        assert result <= 0  # Negated minimum


class TestCEC2010Bounds:
    """Test search space bounds."""

    def test_elliptic_bounds(self):
        """Elliptic should have [-100, 100] bounds."""
        func = SeparableElliptic()
        assert func.default_bounds == (-100.0, 100.0)

    def test_rastrigin_bounds(self):
        """Rastrigin should have [-5, 5] bounds."""
        func = SeparableRastrigin()
        assert func.default_bounds == (-5.0, 5.0)

    def test_ackley_bounds(self):
        """Ackley should have [-32, 32] bounds."""
        func = SeparableAckley()
        assert func.default_bounds == (-32.0, 32.0)

    def test_griewank_bounds(self):
        """Griewank should have [-600, 600] bounds."""
        func = MultiGroupGriewank()
        assert func.default_bounds == (-600.0, 600.0)

    def test_composition_bounds(self):
        """Composition functions should have [-5, 5] bounds."""
        func = Composition1()
        assert func.default_bounds == (-5.0, 5.0)


class TestCEC2010DataLoading:
    """Test that data files load correctly."""

    def test_shift_vector_available(self):
        """Shift vectors should be loadable."""
        func = SeparableElliptic()
        data = func._load_data()
        assert f"shift_{func.func_id}" in data
        assert data[f"shift_{func.func_id}"].shape == (1000,)

    def test_permutation_available(self):
        """Permutation indices should be loadable."""
        func = MultiGroupElliptic()
        perm = func._get_permutation()
        assert perm.shape == (1000,)
        # Check it's a valid permutation (all indices 0-999)
        assert set(perm) == set(range(1000))

    def test_rotation_matrix_available(self):
        """Rotation matrices (50x50) should be loadable."""
        func = MultiGroupElliptic()
        R = func._get_group_rotation(0)
        assert R.shape == (50, 50)
        # Check it's approximately orthogonal
        assert np.allclose(R @ R.T, np.eye(50), atol=1e-10)
