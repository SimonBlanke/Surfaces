"""Property tests for global optima.

These tests verify that functions with known global optima
return the correct value when evaluated at the optimum.
"""

import inspect

import numpy as np
import pytest

import surfaces.test_functions.cec.cec2014 as cec2014
from surfaces.test_functions.algebraic import algebraic_functions
from surfaces.test_functions.bbob import BBOB_FUNCTIONS

from tests.conftest import func_id, instantiate_function

BBOB_FUNCTION_LIST = list(BBOB_FUNCTIONS.values())

CEC2014_FUNCTIONS = [
    v for k, v in vars(cec2014).items()
    if inspect.isclass(v) and not k.startswith("_") and k != "CEC2014Function"
]
CEC2014_UNIMODAL = CEC2014_FUNCTIONS[:3]
CEC2014_MULTIMODAL = CEC2014_FUNCTIONS[3:16]

# =============================================================================
# Algebraic Functions - Global Optima
# =============================================================================


@pytest.mark.algebraic
class TestAlgebraicGlobalOptima:
    """Test global optima for algebraic functions."""

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_has_global_minimum(self, func_class):
        """Functions have f_global attribute."""
        func = instantiate_function(func_class)
        assert hasattr(func, "f_global") or "f_global" in func.spec

    @pytest.mark.parametrize("func_class", algebraic_functions[:10], ids=func_id)
    def test_global_minimum_is_achievable(self, func_class):
        """Evaluating at x_global gives f_global (for subset)."""
        func = instantiate_function(func_class)

        if not hasattr(func, "x_global") and "x_global" not in func.spec:
            pytest.skip("No x_global defined")

        x_global = getattr(func, "x_global", func.spec.get("x_global"))
        f_global = getattr(func, "f_global", func.spec.get("f_global"))

        if x_global is None or f_global is None:
            pytest.skip("Global optimum not defined")

        # Convert to dict format
        if isinstance(x_global, (list, tuple, np.ndarray)):
            params = {f"x{i}": float(v) for i, v in enumerate(x_global)}
        else:
            params = x_global

        result = func(params)
        np.testing.assert_almost_equal(result, f_global, decimal=5)


# =============================================================================
# BBOB Functions - Global Optima
# =============================================================================


@pytest.mark.bbob
class TestBBOBGlobalOptima:
    """Test global optima for BBOB functions."""

    @pytest.mark.parametrize("func_class", BBOB_FUNCTION_LIST, ids=func_id)
    def test_bbob_has_global_minimum(self, func_class):
        """BBOB functions have known global minimum."""
        func = instantiate_function(func_class, n_dim=2)
        assert hasattr(func, "f_global") or "f_global" in func.spec


# =============================================================================
# CEC Functions - Global Optima
# =============================================================================


@pytest.mark.cec
class TestCECGlobalOptima:
    """Test global optima for CEC functions."""

    @pytest.mark.parametrize("func_class", CEC2014_UNIMODAL, ids=func_id)
    def test_cec_unimodal_has_global(self, func_class):
        """CEC unimodal functions have global minimum."""
        func = instantiate_function(func_class, n_dim=10)
        assert hasattr(func, "f_global") or "f_global" in func.spec
