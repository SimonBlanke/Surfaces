# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CEC 2022 Single Objective Bound Constrained benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.benchmark.cec.cec2022 import (
    CEC2022_ALL,
    CEC2022_BASIC,
    CEC2022_HYBRID,
    CEC2022_COMPOSITION,
    ShiftedRotatedZakharov2022,
    ShiftedRotatedRosenbrock2022,
    ShiftedRotatedExpandedSchafferF72022,
    ShiftedRotatedNonContRastrigin2022,
    ShiftedRotatedLevy2022,
    HybridFunction1_2022,
    HybridFunction2_2022,
    HybridFunction3_2022,
    CompositionFunction1_2022,
    CompositionFunction2_2022,
    CompositionFunction3_2022,
    CompositionFunction4_2022,
)


class TestCEC2022FunctionProperties:
    """Test function properties and specs."""

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_has_func_id(self, func_class):
        """Each function must have a func_id."""
        func = func_class()
        assert func.func_id is not None
        assert 1 <= func.func_id <= 12

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_has_f_global(self, func_class):
        """Each function must have f_global defined."""
        func = func_class()
        assert func.f_global is not None
        assert np.isfinite(func.f_global)

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_has_spec(self, func_class):
        """Each function must have specs defined."""
        func = func_class()
        spec = func.spec
        assert "continuous" in spec
        assert "unimodal" in spec
        assert "separable" in spec


class TestCEC2022SupportedDimensions:
    """Test supported dimensions for CEC 2022 functions."""

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_supported_dims(self, func_class):
        """All functions should support dimensions 10, 20."""
        assert func_class.supported_dims == (10, 20)

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    @pytest.mark.parametrize("dim", [10, 20])
    def test_instantiation_all_dims(self, func_class, dim):
        """Functions should instantiate at all supported dimensions."""
        func = func_class(n_dim=dim)
        assert func.n_dim == dim


class TestCEC2022Bounds:
    """Test bounds for CEC 2022 functions."""

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_default_bounds(self, func_class):
        """All functions should have bounds [-100, 100]."""
        func = func_class()
        assert func.default_bounds == (-100.0, 100.0)


class TestCEC2022InputFormats:
    """Test different input formats."""

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_array_input(self, func_class):
        """Function should accept numpy array input."""
        func = func_class()
        result = func(np.zeros(func.n_dim))
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_list_input(self, func_class):
        """Function should accept list input."""
        func = func_class()
        result = func([0.0] * func.n_dim)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_dict_input(self, func_class):
        """Function should accept dict input."""
        func = func_class()
        params = {f"x{i}": 0.0 for i in range(func.n_dim)}
        result = func(params)
        assert np.isfinite(result)


class TestCEC2022DataIntegrity:
    """Test data file integrity."""

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_shift_vector_shape(self, func_class):
        """Shift vectors should have correct shape."""
        func = func_class()
        shift = func._get_shift_vector()
        assert shift.shape == (func.n_dim,)

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_rotation_matrix_shape(self, func_class):
        """Rotation matrices should have correct shape."""
        func = func_class()
        M = func._get_rotation_matrix()
        assert M.shape == (func.n_dim, func.n_dim)

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_data_loaded_correctly(self, func_class):
        """All functions should load their data without error."""
        func = func_class()
        result = func(np.zeros(func.n_dim))
        assert np.isfinite(result)


class TestCEC2022SearchSpace:
    """Test search space properties."""

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_search_space(self, func_class):
        """Search space should have correct dimensions."""
        func = func_class()
        space = func.search_space
        assert len(space) == func.n_dim
        for i in range(func.n_dim):
            assert f"x{i}" in space


class TestCEC2022Objective:
    """Test objective parameter."""

    def test_minimize_objective(self):
        """Minimize objective should return correct values."""
        func = ShiftedRotatedZakharov2022(objective="minimize")
        result = func(func.x_global)
        assert np.isclose(result, func.f_global, rtol=0.1, atol=10.0)

    def test_maximize_objective(self):
        """Maximize objective should negate values."""
        func = ShiftedRotatedZakharov2022(objective="maximize")
        result = func(func.x_global)
        assert result <= 0  # Negated, should be non-positive


class TestCEC2022BatchEvaluation:
    """Test batch evaluation."""

    @pytest.mark.parametrize("func_class", CEC2022_ALL)
    def test_batch_evaluation(self, func_class):
        """Batch evaluation should return correct shape."""
        func = func_class()
        X = np.random.uniform(-100, 100, (5, func.n_dim))
        results = func.batch(X)
        assert results.shape == (5,)
        assert all(np.isfinite(results))


class TestCEC2022FunctionCategories:
    """Test function categories are correct."""

    def test_basic_count(self):
        """Should have 5 basic functions."""
        assert len(CEC2022_BASIC) == 5

    def test_hybrid_count(self):
        """Should have 3 hybrid functions."""
        assert len(CEC2022_HYBRID) == 3

    def test_composition_count(self):
        """Should have 4 composition functions."""
        assert len(CEC2022_COMPOSITION) == 4

    def test_total_count(self):
        """Should have 12 functions total."""
        assert len(CEC2022_ALL) == 12


class TestCEC2022UnimodalProperty:
    """Test unimodal property is correctly set."""

    def test_zakharov_unimodal(self):
        """F1 Zakharov should be unimodal."""
        func = ShiftedRotatedZakharov2022()
        assert func.spec["unimodal"] is True

    @pytest.mark.parametrize("func_class", CEC2022_ALL[1:])  # All except F1
    def test_multimodal_functions(self, func_class):
        """F2-F12 should be multimodal."""
        func = func_class()
        assert func.spec["unimodal"] is False


class TestCEC2022NonContinuous:
    """Test non-continuous function property."""

    def test_non_cont_rastrigin_not_continuous(self):
        """F4 Non-Continuous Rastrigin should be marked as non-continuous."""
        func = ShiftedRotatedNonContRastrigin2022()
        assert func.spec["continuous"] is False


class TestCEC2022FBiasValues:
    """Test f_bias values match specification."""

    def test_f1_f_global(self):
        """F1 Zakharov should have f_global=300."""
        func = ShiftedRotatedZakharov2022()
        assert func.f_global == 300.0

    def test_f2_f_global(self):
        """F2 Rosenbrock should have f_global=400."""
        func = ShiftedRotatedRosenbrock2022()
        assert func.f_global == 400.0

    def test_f3_f_global(self):
        """F3 Expanded Schaffer F7 should have f_global=600."""
        func = ShiftedRotatedExpandedSchafferF72022()
        assert func.f_global == 600.0

    def test_f4_f_global(self):
        """F4 Non-Continuous Rastrigin should have f_global=800."""
        func = ShiftedRotatedNonContRastrigin2022()
        assert func.f_global == 800.0

    def test_f5_f_global(self):
        """F5 Levy should have f_global=900."""
        func = ShiftedRotatedLevy2022()
        assert func.f_global == 900.0

    def test_f6_f_global(self):
        """F6 Hybrid 1 should have f_global=1800."""
        func = HybridFunction1_2022()
        assert func.f_global == 1800.0

    def test_f7_f_global(self):
        """F7 Hybrid 2 should have f_global=2000."""
        func = HybridFunction2_2022()
        assert func.f_global == 2000.0

    def test_f8_f_global(self):
        """F8 Hybrid 3 should have f_global=2200."""
        func = HybridFunction3_2022()
        assert func.f_global == 2200.0

    def test_f9_f_global(self):
        """F9 Composition 1 should have f_global=2300."""
        func = CompositionFunction1_2022()
        assert func.f_global == 2300.0

    def test_f10_f_global(self):
        """F10 Composition 2 should have f_global=2400."""
        func = CompositionFunction2_2022()
        assert func.f_global == 2400.0

    def test_f11_f_global(self):
        """F11 Composition 3 should have f_global=2600."""
        func = CompositionFunction3_2022()
        assert func.f_global == 2600.0

    def test_f12_f_global(self):
        """F12 Composition 4 should have f_global=2700."""
        func = CompositionFunction4_2022()
        assert func.f_global == 2700.0
