# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Property tests for function spec consistency.

These tests verify that function specifications (_spec) are
properly defined and consistent across the class hierarchy.
"""

import pytest

from surfaces.test_functions.algebraic import algebraic_functions
from surfaces.test_functions.bbob import BBOB_FUNCTIONS
from surfaces.test_functions.cec.cec2014 import CEC2014_FUNCTIONS

from tests.conftest import func_id, instantiate_function

BBOB_FUNCTION_LIST = list(BBOB_FUNCTIONS.values())
CEC2014_UNIMODAL = CEC2014_FUNCTIONS[:3]

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
        assert isinstance(spec, dict)

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_algebraic_spec_has_required_keys(self, func_class):
        """Algebraic function specs have required keys."""
        func = instantiate_function(func_class)
        spec = func.spec
        required_keys = ["continuous", "differentiable", "default_bounds"]
        for key in required_keys:
            assert key in spec, f"Spec missing required key: {key}"

    @pytest.mark.parametrize("func_class", BBOB_FUNCTION_LIST, ids=func_id)
    def test_bbob_has_func_id(self, func_class):
        """BBOB functions have func_id in spec."""
        func = instantiate_function(func_class, n_dim=2)
        assert func.spec.get("func_id") is not None
        assert 1 <= func.spec["func_id"] <= 24

    @pytest.mark.skipif(not HAS_CEC2014, reason="CEC 2014 data not installed")
    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS, ids=func_id)
    def test_cec2014_has_func_id(self, func_class):
        """CEC 2014 functions have func_id."""
        func = instantiate_function(func_class, n_dim=10)
        assert func.func_id is not None
        assert 1 <= func.func_id <= 30


# =============================================================================
# Spec Value Consistency
# =============================================================================


class TestSpecValues:
    """Test that spec values are consistent."""

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_continuous_is_boolean(self, func_class):
        """continuous spec value is boolean."""
        func = instantiate_function(func_class)
        assert isinstance(func.spec.get("continuous", True), bool)

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_differentiable_is_boolean(self, func_class):
        """differentiable spec value is boolean."""
        func = instantiate_function(func_class)
        assert isinstance(func.spec.get("differentiable", True), bool)

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_convex_is_boolean(self, func_class):
        """convex spec value is boolean."""
        func = instantiate_function(func_class)
        assert isinstance(func.spec.get("convex", False), bool)

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_separable_is_boolean(self, func_class):
        """separable spec value is boolean."""
        func = instantiate_function(func_class)
        assert isinstance(func.spec.get("separable", False), bool)

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_unimodal_is_boolean(self, func_class):
        """unimodal spec value is boolean."""
        func = instantiate_function(func_class)
        assert isinstance(func.spec.get("unimodal", False), bool)


# =============================================================================
# CEC 2014 Spec Consistency
# =============================================================================


@pytest.mark.cec
@pytest.mark.cec2014
class TestCEC2014Specs:
    """Test CEC 2014 function spec consistency."""

    @pytest.mark.skipif(not HAS_CEC2014, reason="CEC 2014 data not installed")
    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS, ids=func_id)
    def test_scalable_is_true(self, func_class):
        """All CEC 2014 functions are scalable."""
        func = instantiate_function(func_class, n_dim=10)
        assert func.spec["scalable"] is True

    @pytest.mark.skipif(not HAS_CEC2014, reason="CEC 2014 data not installed")
    @pytest.mark.parametrize("func_class", CEC2014_UNIMODAL, ids=func_id)
    def test_unimodal_functions_marked(self, func_class):
        """Unimodal functions (F1-F3) have unimodal=True."""
        func = instantiate_function(func_class, n_dim=10)
        assert func.spec["unimodal"] is True

    @pytest.mark.skipif(not HAS_CEC2014, reason="CEC 2014 data not installed")
    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS[3:], ids=func_id)
    def test_multimodal_functions_marked(self, func_class):
        """Multimodal functions (F4-F30) have unimodal=False."""
        func = instantiate_function(func_class, n_dim=10)
        assert func.spec["unimodal"] is False


# =============================================================================
# BBOB Spec Consistency
# =============================================================================


@pytest.mark.bbob
class TestBBOBSpecs:
    """Test BBOB function spec consistency."""

    @pytest.mark.parametrize("func_class", BBOB_FUNCTION_LIST, ids=func_id)
    def test_bbob_has_name(self, func_class):
        """BBOB functions have name in spec."""
        func = instantiate_function(func_class, n_dim=2)
        name = func.spec.get("name")
        assert name is not None and name != ""

    @pytest.mark.parametrize("func_class", BBOB_FUNCTION_LIST, ids=func_id)
    def test_bbob_scalable(self, func_class):
        """BBOB functions are scalable."""
        func = instantiate_function(func_class, n_dim=2)
        assert func.spec.get("scalable", True) is True
