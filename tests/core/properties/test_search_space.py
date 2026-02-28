"""Property tests for search space definitions (core functions).

These tests verify that algebraic, engineering, and BBOB functions
have valid search spaces with correct structure and bounds.
"""

import pytest

from surfaces.test_functions.algebraic import algebraic_functions
from surfaces.test_functions.algebraic.constrained import constrained_functions
from surfaces.test_functions.benchmark.bbob import bbob_functions
from tests.conftest import func_id, instantiate_function


class TestSearchSpaceStructure:
    """Test that search spaces have correct structure."""

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_algebraic_search_space_is_dict(self, func_class):
        """Algebraic functions have dict search space."""
        func = instantiate_function(func_class)
        assert isinstance(func.search_space, dict)
        assert len(func.search_space) > 0

    @pytest.mark.parametrize("func_class", constrained_functions, ids=func_id)
    def test_engineering_search_space_is_dict(self, func_class):
        """Engineering functions have dict search space."""
        func = instantiate_function(func_class)
        assert isinstance(func.search_space, dict)
        assert len(func.search_space) > 0

    @pytest.mark.parametrize("func_class", bbob_functions, ids=func_id)
    def test_bbob_search_space_is_dict(self, func_class):
        """BBOB functions have dict search space."""
        func = instantiate_function(func_class, n_dim=2)
        assert isinstance(func.search_space, dict)
        assert len(func.search_space) == 2


class TestSearchSpaceKeys:
    """Test that search space keys are valid."""

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_algebraic_keys_are_strings(self, func_class):
        """Algebraic search space keys are strings."""
        func = instantiate_function(func_class)
        for key in func.search_space.keys():
            assert isinstance(key, str)

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_algebraic_keys_follow_pattern(self, func_class):
        """Algebraic search space keys follow x0, x1, ... pattern."""
        func = instantiate_function(func_class)
        keys = list(func.search_space.keys())
        expected = [f"x{i}" for i in range(len(keys))]
        assert keys == expected


class TestSearchSpaceValues:
    """Test that search space values are valid."""

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_algebraic_values_are_iterable(self, func_class):
        """Algebraic search space values are iterable."""
        func = instantiate_function(func_class)
        for values in func.search_space.values():
            assert hasattr(values, "__iter__")

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_algebraic_values_nonempty(self, func_class):
        """Algebraic search space values are non-empty."""
        func = instantiate_function(func_class)
        for values in func.search_space.values():
            assert len(list(values)) > 0
