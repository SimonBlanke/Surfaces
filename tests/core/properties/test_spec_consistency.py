"""Property tests for function spec consistency (core functions).

These tests verify that function specifications (_spec) are
properly defined and consistent across algebraic and BBOB functions.
"""

import pytest

from surfaces.test_functions.algebraic import algebraic_functions
from surfaces.test_functions.benchmark.bbob import bbob_functions
from tests.conftest import func_id, instantiate_function

# =============================================================================
# Spec Existence and Structure
# =============================================================================


class TestSpecStructure:
    """Test that specs have correct structure."""

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_algebraic_has_spec(self, func_class):
        """Algebraic functions have spec property."""
        func = instantiate_function(func_class)
        spec = func.spec
        assert isinstance(spec.as_dict(), dict)

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_algebraic_spec_has_required_keys(self, func_class):
        """Algebraic function specs have required keys."""
        func = instantiate_function(func_class)
        spec = func.spec
        required_keys = ["continuous", "differentiable", "default_bounds"]
        for key in required_keys:
            assert key in spec, f"Spec missing required key: {key}"

    @pytest.mark.parametrize("func_class", bbob_functions, ids=func_id)
    def test_bbob_has_func_id(self, func_class):
        """BBOB functions have func_id in spec."""
        func = instantiate_function(func_class, n_dim=2)
        assert "func_id" in func.spec
        assert isinstance(func.spec["func_id"], int)


# =============================================================================
# Spec Consistency
# =============================================================================


class TestSpecConsistency:
    """Test that specs are consistent across instances."""

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_spec_immutable_across_calls(self, func_class):
        """Spec doesn't change between calls."""
        func = instantiate_function(func_class)
        spec1 = func.spec

        params = {key: 0.5 for key in func.search_space.keys()}
        _ = func(params)

        spec2 = func.spec
        assert spec1.as_dict() == spec2.as_dict()

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_spec_same_for_different_instances(self, func_class):
        """Spec is identical for different instances of same class."""
        func1 = instantiate_function(func_class)
        func2 = instantiate_function(func_class)
        assert func1.spec.as_dict() == func2.spec.as_dict()


# =============================================================================
# Spec Values
# =============================================================================


class TestSpecValues:
    """Test that spec values are valid."""

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_continuous_is_bool(self, func_class):
        """Continuous field is boolean."""
        func = instantiate_function(func_class)
        assert isinstance(func.spec.get("continuous"), bool)

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_differentiable_is_bool(self, func_class):
        """Differentiable field is boolean."""
        func = instantiate_function(func_class)
        assert isinstance(func.spec.get("differentiable"), bool)

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_default_bounds_format(self, func_class):
        """Default bounds are tuple of two numbers."""
        func = instantiate_function(func_class)
        bounds = func.spec.get("default_bounds")
        assert isinstance(bounds, (list, tuple))
        assert len(bounds) == 2
        assert bounds[0] < bounds[1]
