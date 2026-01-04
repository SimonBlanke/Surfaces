# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Property tests for search space definitions.

These tests verify that all functions have valid search spaces
with correct structure and bounds.
"""

import pytest

from surfaces.test_functions.algebraic import algebraic_functions
from surfaces.test_functions.bbob import BBOB_FUNCTIONS
from surfaces.test_functions.cec.cec2014 import CEC2014_FUNCTIONS
from surfaces.test_functions.engineering import engineering_functions
from surfaces.test_functions.machine_learning import machine_learning_functions

from tests.conftest import func_id, instantiate_function

BBOB_FUNCTION_LIST = list(BBOB_FUNCTIONS.values())

# =============================================================================
# Search Space Structure
# =============================================================================


class TestSearchSpaceStructure:
    """Test that search spaces have correct structure."""

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_algebraic_search_space_is_dict(self, func_class):
        """Algebraic function search spaces are dictionaries."""
        func = instantiate_function(func_class)
        assert isinstance(func.search_space, dict)
        assert len(func.search_space) > 0

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_algebraic_search_space_keys(self, func_class):
        """Algebraic function search space keys follow x0, x1, ... pattern."""
        func = instantiate_function(func_class)
        for i, key in enumerate(sorted(func.search_space.keys())):
            assert key == f"x{i}", f"Expected x{i}, got {key}"

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_algebraic_search_space_values_iterable(self, func_class):
        """Algebraic function search space values are iterable."""
        func = instantiate_function(func_class)
        for key, values in func.search_space.items():
            assert hasattr(values, "__iter__"), f"{key} values must be iterable"
            values_list = list(values)
            assert len(values_list) > 0, f"{key} must have at least one value"

    @pytest.mark.parametrize("func_class", engineering_functions, ids=func_id)
    def test_engineering_search_space_is_dict(self, func_class):
        """Engineering function search spaces are dictionaries."""
        func = instantiate_function(func_class)
        assert isinstance(func.search_space, dict)
        assert len(func.search_space) > 0


# =============================================================================
# Default Bounds
# =============================================================================


class TestDefaultBounds:
    """Test that default bounds are properly defined."""

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_algebraic_has_default_bounds(self, func_class):
        """Algebraic functions define default bounds."""
        func = instantiate_function(func_class)
        bounds = func.default_bounds
        assert bounds is not None
        assert len(bounds) == 2
        assert bounds[0] < bounds[1], "Lower bound must be less than upper bound"

    @pytest.mark.parametrize("func_class", BBOB_FUNCTION_LIST, ids=func_id)
    def test_bbob_default_bounds(self, func_class):
        """BBOB functions have [-5, 5] default bounds."""
        func = instantiate_function(func_class, n_dim=2)
        assert func.default_bounds == (-5.0, 5.0)

    @pytest.mark.skipif(not HAS_CEC2014, reason="CEC 2014 data not installed")
    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS, ids=func_id)
    def test_cec2014_default_bounds(self, func_class):
        """CEC 2014 functions have [-100, 100] default bounds."""
        func = instantiate_function(func_class, n_dim=10)
        assert func.default_bounds == (-100.0, 100.0)


# =============================================================================
# Dimension Handling
# =============================================================================


class TestDimensionHandling:
    """Test dimension-related properties."""

    @pytest.mark.parametrize("func_class", BBOB_FUNCTION_LIST, ids=func_id)
    @pytest.mark.parametrize("n_dim", [2, 5, 10, 20])
    def test_bbob_dimension_scaling(self, func_class, n_dim):
        """BBOB functions scale to different dimensions."""
        func = instantiate_function(func_class, n_dim=n_dim)
        assert len(func.search_space) == n_dim

    @pytest.mark.skipif(not HAS_CEC2014, reason="CEC 2014 data not installed")
    @pytest.mark.parametrize("n_dim", [10, 20, 30, 50, 100])
    def test_cec2014_supported_dimensions(self, n_dim):
        """CEC 2014 functions support standard dimensions."""
        from surfaces.test_functions.cec.cec2014 import RotatedHighConditionedElliptic

        func = RotatedHighConditionedElliptic(n_dim=n_dim)
        assert func.n_dim == n_dim
        assert len(func.search_space) == n_dim


# =============================================================================
# ML Function Search Spaces
# =============================================================================


@pytest.mark.ml
class TestMLSearchSpaces:
    """Test ML function search space properties."""

    @pytest.mark.skipif(not HAS_ML, reason="scikit-learn not installed")
    @pytest.mark.parametrize("func_class", machine_learning_functions, ids=func_id)
    def test_ml_search_space_has_hyperparameters(self, func_class):
        """ML functions have hyperparameter search spaces."""
        func = instantiate_function(func_class)
        assert len(func.search_space) > 0
        # ML functions typically have at least one hyperparameter
        # plus cv and dataset parameters in defaults

    @pytest.mark.skipif(not HAS_ML, reason="scikit-learn not installed")
    @pytest.mark.parametrize("func_class", machine_learning_functions, ids=func_id)
    def test_ml_search_space_values_valid(self, func_class):
        """ML function search space values are valid."""
        func = instantiate_function(func_class)
        for key, values in func.search_space.items():
            if hasattr(values, "__iter__") and not isinstance(values, str):
                values_list = list(values)
                assert len(values_list) > 0, f"{key} must have at least one value"
