# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CEC 2020 Single Objective Bound Constrained benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.benchmark.cec.cec2020 import (
    CEC2020_ALL,
    CEC2020_BASIC,
    CEC2020_HYBRID,
    CEC2020_COMPOSITION,
    ShiftedRotatedBentCigar2020,
    ShiftedRotatedSchwefel2020,
    ShiftedRotatedLunacekBiRastrigin2020,
    ExpandedGriewankRosenbrock2020,
    HybridFunction1_2020,
    HybridFunction2_2020,
    HybridFunction3_2020,
    CompositionFunction1_2020,
    CompositionFunction2_2020,
    CompositionFunction3_2020,
)


class TestCEC2020FunctionProperties:
    """Test function properties and specs."""

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_has_func_id(self, func_class):
        """Each function must have a func_id."""
        func = func_class()
        assert func.func_id is not None
        assert 1 <= func.func_id <= 10

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_has_f_global(self, func_class):
        """Each function must have f_global defined."""
        func = func_class()
        assert func.f_global is not None
        assert np.isfinite(func.f_global)

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_has_spec(self, func_class):
        """Each function must have specs defined."""
        func = func_class()
        spec = func.spec
        assert "continuous" in spec
        assert "unimodal" in spec
        assert "separable" in spec


class TestCEC2020SupportedDimensions:
    """Test supported dimensions for CEC 2020 functions."""

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_supported_dims(self, func_class):
        """All functions should support dimensions 5, 10, 15, 20."""
        assert func_class.supported_dims == (5, 10, 15, 20)

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    @pytest.mark.parametrize("dim", [5, 10, 15, 20])
    def test_instantiation_all_dims(self, func_class, dim):
        """Functions should instantiate at all supported dimensions."""
        func = func_class(n_dim=dim)
        assert func.n_dim == dim


class TestCEC2020Bounds:
    """Test bounds for CEC 2020 functions."""

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_default_bounds(self, func_class):
        """All functions should have bounds [-100, 100]."""
        func = func_class()
        assert func.default_bounds == (-100.0, 100.0)


class TestCEC2020InputFormats:
    """Test different input formats."""

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_array_input(self, func_class):
        """Function should accept numpy array input."""
        func = func_class()
        result = func(np.zeros(func.n_dim))
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_list_input(self, func_class):
        """Function should accept list input."""
        func = func_class()
        result = func([0.0] * func.n_dim)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_dict_input(self, func_class):
        """Function should accept dict input."""
        func = func_class()
        params = {f"x{i}": 0.0 for i in range(func.n_dim)}
        result = func(params)
        assert np.isfinite(result)


class TestCEC2020DataIntegrity:
    """Test data file integrity."""

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_shift_vector_shape(self, func_class):
        """Shift vectors should have correct shape."""
        func = func_class()
        shift = func._get_shift_vector()
        assert shift.shape == (func.n_dim,)

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_rotation_matrix_shape(self, func_class):
        """Rotation matrices should have correct shape."""
        func = func_class()
        M = func._get_rotation_matrix()
        assert M.shape == (func.n_dim, func.n_dim)

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_data_loaded_correctly(self, func_class):
        """All functions should load their data without error."""
        func = func_class()
        result = func(np.zeros(func.n_dim))
        assert np.isfinite(result)


class TestCEC2020SearchSpace:
    """Test search space properties."""

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_search_space(self, func_class):
        """Search space should have correct dimensions."""
        func = func_class()
        space = func.search_space
        assert len(space) == func.n_dim
        for i in range(func.n_dim):
            assert f"x{i}" in space


class TestCEC2020Objective:
    """Test objective parameter."""

    def test_minimize_objective(self):
        """Minimize objective should return correct values."""
        func = ShiftedRotatedBentCigar2020(objective="minimize")
        result = func(func.x_global)
        assert np.isclose(result, func.f_global, rtol=0.1, atol=10.0)

    def test_maximize_objective(self):
        """Maximize objective should negate values."""
        func = ShiftedRotatedBentCigar2020(objective="maximize")
        result = func(func.x_global)
        assert result <= 0  # Negated, should be non-positive


class TestCEC2020BatchEvaluation:
    """Test batch evaluation."""

    @pytest.mark.parametrize("func_class", CEC2020_ALL)
    def test_batch_evaluation(self, func_class):
        """Batch evaluation should return correct shape."""
        func = func_class()
        X = np.random.uniform(-100, 100, (5, func.n_dim))
        results = func.batch(X)
        assert results.shape == (5,)
        assert all(np.isfinite(results))


class TestCEC2020FunctionCategories:
    """Test function categories are correct."""

    def test_basic_count(self):
        """Should have 4 basic functions."""
        assert len(CEC2020_BASIC) == 4

    def test_hybrid_count(self):
        """Should have 3 hybrid functions."""
        assert len(CEC2020_HYBRID) == 3

    def test_composition_count(self):
        """Should have 3 composition functions."""
        assert len(CEC2020_COMPOSITION) == 3

    def test_total_count(self):
        """Should have 10 functions total."""
        assert len(CEC2020_ALL) == 10


class TestCEC2020UnimodalProperty:
    """Test unimodal property is correctly set."""

    def test_bent_cigar_unimodal(self):
        """F1 Bent Cigar should be unimodal."""
        func = ShiftedRotatedBentCigar2020()
        assert func.spec["unimodal"] is True

    @pytest.mark.parametrize("func_class", CEC2020_ALL[1:])  # All except F1
    def test_multimodal_functions(self, func_class):
        """F2-F10 should be multimodal."""
        func = func_class()
        assert func.spec["unimodal"] is False
