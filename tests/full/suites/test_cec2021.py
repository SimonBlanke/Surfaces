# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CEC 2021 Single Objective Bound Constrained benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.benchmark.cec.cec2021 import (
    CEC2021_ALL,
    CEC2021_BASIC,
    CEC2021_HYBRID,
    CEC2021_COMPOSITION,
    ShiftedRotatedBentCigar2021,
    ShiftedRotatedSchwefel2021,
    ShiftedRotatedLunacekBiRastrigin2021,
    ExpandedGriewankRosenbrock2021,
    HybridFunction1_2021,
    HybridFunction2_2021,
    HybridFunction3_2021,
    CompositionFunction1_2021,
    CompositionFunction2_2021,
    CompositionFunction3_2021,
)


class TestCEC2021FunctionProperties:
    """Test function properties and specs."""

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_has_func_id(self, func_class):
        """Each function must have a func_id."""
        func = func_class()
        assert func.func_id is not None
        assert 1 <= func.func_id <= 10

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_has_f_global(self, func_class):
        """Each function must have f_global defined."""
        func = func_class()
        assert func.f_global is not None
        assert np.isfinite(func.f_global)

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_has_spec(self, func_class):
        """Each function must have specs defined."""
        func = func_class()
        spec = func.spec
        assert "continuous" in spec
        assert "unimodal" in spec
        assert "separable" in spec


class TestCEC2021SupportedDimensions:
    """Test supported dimensions for CEC 2021 functions."""

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_supported_dims(self, func_class):
        """All functions should support dimensions 10, 20."""
        assert func_class.supported_dims == (10, 20)

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    @pytest.mark.parametrize("dim", [10, 20])
    def test_instantiation_all_dims(self, func_class, dim):
        """Functions should instantiate at all supported dimensions."""
        func = func_class(n_dim=dim)
        assert func.n_dim == dim


class TestCEC2021Bounds:
    """Test bounds for CEC 2021 functions."""

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_default_bounds(self, func_class):
        """All functions should have bounds [-100, 100]."""
        func = func_class()
        assert func.default_bounds == (-100.0, 100.0)


class TestCEC2021InputFormats:
    """Test different input formats."""

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_array_input(self, func_class):
        """Function should accept numpy array input."""
        func = func_class()
        result = func(np.zeros(func.n_dim))
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_list_input(self, func_class):
        """Function should accept list input."""
        func = func_class()
        result = func([0.0] * func.n_dim)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_dict_input(self, func_class):
        """Function should accept dict input."""
        func = func_class()
        params = {f"x{i}": 0.0 for i in range(func.n_dim)}
        result = func(params)
        assert np.isfinite(result)


class TestCEC2021DataIntegrity:
    """Test data file integrity."""

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_shift_vector_shape(self, func_class):
        """Shift vectors should have correct shape."""
        func = func_class()
        shift = func._get_shift_vector()
        assert shift.shape == (func.n_dim,)

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_rotation_matrix_shape(self, func_class):
        """Rotation matrices should have correct shape."""
        func = func_class()
        M = func._get_rotation_matrix()
        assert M.shape == (func.n_dim, func.n_dim)

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_data_loaded_correctly(self, func_class):
        """All functions should load their data without error."""
        func = func_class()
        result = func(np.zeros(func.n_dim))
        assert np.isfinite(result)


class TestCEC2021SearchSpace:
    """Test search space properties."""

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_search_space(self, func_class):
        """Search space should have correct dimensions."""
        func = func_class()
        space = func.search_space
        assert len(space) == func.n_dim
        for i in range(func.n_dim):
            assert f"x{i}" in space


class TestCEC2021Objective:
    """Test objective parameter."""

    def test_minimize_objective(self):
        """Minimize objective should return correct values."""
        func = ShiftedRotatedBentCigar2021(objective="minimize")
        result = func(func.x_global)
        assert np.isclose(result, func.f_global, rtol=0.1, atol=10.0)

    def test_maximize_objective(self):
        """Maximize objective should negate values."""
        func = ShiftedRotatedBentCigar2021(objective="maximize")
        result = func(func.x_global)
        assert result <= 0  # Negated, should be non-positive


class TestCEC2021BatchEvaluation:
    """Test batch evaluation."""

    @pytest.mark.parametrize("func_class", CEC2021_ALL)
    def test_batch_evaluation(self, func_class):
        """Batch evaluation should return correct shape."""
        func = func_class()
        X = np.random.uniform(-100, 100, (5, func.n_dim))
        results = func.batch(X)
        assert results.shape == (5,)
        assert all(np.isfinite(results))


class TestCEC2021FunctionCategories:
    """Test function categories are correct."""

    def test_basic_count(self):
        """Should have 4 basic functions."""
        assert len(CEC2021_BASIC) == 4

    def test_hybrid_count(self):
        """Should have 3 hybrid functions."""
        assert len(CEC2021_HYBRID) == 3

    def test_composition_count(self):
        """Should have 3 composition functions."""
        assert len(CEC2021_COMPOSITION) == 3

    def test_total_count(self):
        """Should have 10 functions total."""
        assert len(CEC2021_ALL) == 10


class TestCEC2021UnimodalProperty:
    """Test unimodal property is correctly set."""

    def test_bent_cigar_unimodal(self):
        """F1 Bent Cigar should be unimodal."""
        func = ShiftedRotatedBentCigar2021()
        assert func.spec["unimodal"] is True

    @pytest.mark.parametrize("func_class", CEC2021_ALL[1:])  # All except F1
    def test_multimodal_functions(self, func_class):
        """F2-F10 should be multimodal."""
        func = func_class()
        assert func.spec["unimodal"] is False


class TestCEC2021IdenticalToCEC2020:
    """Test that CEC 2021 has same structure as CEC 2020."""

    def test_same_f_bias_values(self):
        """CEC 2021 should have same f_bias values as CEC 2020."""
        from surfaces.test_functions.benchmark.cec.cec2020 import CEC2020_ALL

        for i, (cls2020, cls2021) in enumerate(zip(CEC2020_ALL, CEC2021_ALL)):
            func2020 = cls2020(n_dim=10)
            func2021 = cls2021(n_dim=10)
            assert func2020.f_global == func2021.f_global, f"F{i+1} f_global mismatch"
