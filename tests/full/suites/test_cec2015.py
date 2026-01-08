# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CEC 2015 benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.benchmark.cec.cec2015 import (
    CEC2015_ALL,
    CEC2015_COMPOSITION,
    CEC2015_HYBRID,
    CEC2015_MULTIMODAL,
    CEC2015_UNIMODAL,
    CompositionFunction1_2015,
    CompositionFunction2_2015,
    CompositionFunction3_2015,
    ExpandedGriewankRosenbrock2015,
    ExpandedScafferF62015,
    HybridFunction1_2015,
    HybridFunction2_2015,
    HybridFunction3_2015,
    RotatedBentCigar2015,
    RotatedDiscus2015,
    ShiftedRotatedHappyCat2015,
    ShiftedRotatedHGBat2015,
    ShiftedRotatedKatsuura2015,
    ShiftedRotatedSchwefel2015,
    ShiftedRotatedWeierstrass2015,
)


class TestCEC2015FunctionProperties:
    """Test function properties and specs."""

    @pytest.mark.parametrize("func_class", CEC2015_ALL)
    def test_has_func_id(self, func_class):
        """Each function must have a func_id."""
        func = func_class(n_dim=10)
        assert func.func_id is not None
        assert 1 <= func.func_id <= 15

    @pytest.mark.parametrize("func_class", CEC2015_ALL)
    def test_f_global_matches_func_id(self, func_class):
        """f_global should be func_id * 100."""
        func = func_class(n_dim=10)
        assert func.f_global == func.func_id * 100

    @pytest.mark.parametrize("func_class", CEC2015_ALL)
    def test_has_spec(self, func_class):
        """Each function must have specs defined."""
        func = func_class(n_dim=10)
        spec = func.spec
        assert "continuous" in spec
        assert "scalable" in spec

    @pytest.mark.parametrize("func_class", CEC2015_UNIMODAL)
    def test_unimodal_spec(self, func_class):
        """Unimodal functions (F1-F2) should have unimodal=True."""
        func = func_class(n_dim=10)
        assert func.spec["unimodal"] is True

    @pytest.mark.parametrize("func_class", CEC2015_MULTIMODAL + CEC2015_HYBRID + CEC2015_COMPOSITION)
    def test_multimodal_spec(self, func_class):
        """Multimodal functions (F3-F15) should have unimodal=False."""
        func = func_class(n_dim=10)
        assert func.spec["unimodal"] is False


class TestCEC2015Dimensions:
    """Test dimension handling."""

    @pytest.mark.parametrize("dim", [10, 30])
    def test_supported_dimensions(self, dim):
        """Functions should work with supported dimensions (10, 30)."""
        func = RotatedBentCigar2015(n_dim=dim)
        assert func.n_dim == dim
        result = func(np.zeros(dim))
        assert np.isfinite(result)

    def test_unsupported_dimension_raises(self):
        """Unsupported dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="n_dim must be one of"):
            RotatedBentCigar2015(n_dim=15)

    @pytest.mark.parametrize("func_class", CEC2015_ALL)
    @pytest.mark.parametrize("dim", [10, 30])
    def test_all_functions_all_dims(self, func_class, dim):
        """All functions should work with all supported dimensions."""
        func = func_class(n_dim=dim)
        result = func(np.zeros(dim))
        assert np.isfinite(result)


class TestCEC2015InputFormats:
    """Test different input formats."""

    def test_array_input(self):
        """Function should accept numpy array input."""
        func = RotatedBentCigar2015(n_dim=10)
        result = func(np.zeros(10))
        assert np.isfinite(result)

    def test_list_input(self):
        """Function should accept list input."""
        func = RotatedBentCigar2015(n_dim=10)
        result = func([0.0] * 10)
        assert np.isfinite(result)

    def test_dict_input(self):
        """Function should accept dict input."""
        func = RotatedBentCigar2015(n_dim=10)
        params = {f"x{i}": 0.0 for i in range(10)}
        result = func(params)
        assert np.isfinite(result)


class TestCEC2015DataIntegrity:
    """Test data file integrity."""

    def test_rotation_matrix_shape(self):
        """Rotation matrices should have correct shape."""
        func = RotatedBentCigar2015(n_dim=10)
        M = func._get_rotation_matrix()
        assert M.shape == (10, 10)

    def test_shift_vector_in_bounds(self):
        """Shift vectors should be within [-80, 80] (inside search bounds)."""
        func = RotatedBentCigar2015(n_dim=10)
        shift = func._get_shift_vector()
        assert np.all(np.abs(shift) <= 80)

    @pytest.mark.parametrize("func_class", CEC2015_ALL)
    def test_data_loaded_correctly(self, func_class):
        """All functions should load their data without error."""
        func = func_class(n_dim=10)
        result = func(np.zeros(10))
        assert np.isfinite(result)


class TestCEC2015SearchSpace:
    """Test search space properties."""

    def test_default_bounds(self):
        """Default bounds should be [-100, 100]."""
        func = RotatedBentCigar2015(n_dim=10)
        assert func.default_bounds == (-100.0, 100.0)

    def test_search_space(self):
        """Search space should have correct dimensions."""
        func = RotatedBentCigar2015(n_dim=10)
        space = func.search_space
        assert len(space) == 10
        for i in range(10):
            assert f"x{i}" in space


class TestCEC2015Objective:
    """Test objective parameter."""

    def test_minimize_objective(self):
        """Minimize objective should return positive-biased values."""
        func = RotatedBentCigar2015(n_dim=10, objective="minimize")
        result = func(func.x_global)
        assert result == func.f_global

    def test_maximize_objective(self):
        """Maximize objective should negate values."""
        func = RotatedBentCigar2015(n_dim=10, objective="maximize")
        result = func(func.x_global)
        assert result == -func.f_global


class TestCEC2015GlobalOptimum:
    """Test that f(x_global) = f_global for unimodal functions."""

    @pytest.mark.parametrize("func_class", CEC2015_UNIMODAL)
    def test_global_optimum_unimodal(self, func_class):
        """Test global optimum for unimodal functions (F1-F2)."""
        func = func_class(n_dim=10)
        result = func(func.x_global)
        assert np.isclose(
            result, func.f_global, rtol=1e-6
        ), f"{func.name}: f(x_global)={result}, expected {func.f_global}"


class TestCEC2015FunctionCategories:
    """Test function categories are correct."""

    def test_unimodal_count(self):
        """Should have 2 unimodal functions."""
        assert len(CEC2015_UNIMODAL) == 2

    def test_multimodal_count(self):
        """Should have 7 multimodal functions."""
        assert len(CEC2015_MULTIMODAL) == 7

    def test_hybrid_count(self):
        """Should have 3 hybrid functions."""
        assert len(CEC2015_HYBRID) == 3

    def test_composition_count(self):
        """Should have 3 composition functions."""
        assert len(CEC2015_COMPOSITION) == 3

    def test_total_count(self):
        """Should have 15 functions total."""
        assert len(CEC2015_ALL) == 15


class TestCEC2015HybridFunctions:
    """Test hybrid functions specifically."""

    @pytest.mark.parametrize("func_class", CEC2015_HYBRID)
    def test_hybrid_has_shuffle(self, func_class):
        """Hybrid functions should have shuffle indices."""
        func = func_class(n_dim=10)
        shuffle = func._get_shuffle_indices()
        assert len(shuffle) == 10
        assert set(shuffle) == set(range(10))

    @pytest.mark.parametrize("func_class", CEC2015_HYBRID)
    def test_hybrid_finite_output(self, func_class):
        """Hybrid functions should produce finite output."""
        func = func_class(n_dim=10)
        result = func(np.random.randn(10))
        assert np.isfinite(result)


class TestCEC2015CompositionFunctions:
    """Test composition functions specifically."""

    @pytest.mark.parametrize("func_class", CEC2015_COMPOSITION)
    def test_composition_finite_output(self, func_class):
        """Composition functions should produce finite output."""
        func = func_class(n_dim=10)
        result = func(np.random.randn(10))
        assert np.isfinite(result)

    def test_composition_components(self):
        """Composition functions should have correct number of components."""
        func1 = CompositionFunction1_2015(n_dim=10)
        func2 = CompositionFunction2_2015(n_dim=10)
        func3 = CompositionFunction3_2015(n_dim=10)

        assert func1._n_components == 5
        assert func2._n_components == 3
        assert func3._n_components == 5
