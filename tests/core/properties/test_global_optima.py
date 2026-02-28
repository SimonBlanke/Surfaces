"""Property tests for global optima (core functions).

These tests verify that functions with known global optima
return the correct value when evaluated at the optimum.
"""

import numpy as np
import pytest

from surfaces.test_functions.algebraic import algebraic_functions
from surfaces.test_functions.benchmark.bbob import bbob_functions
from tests.conftest import func_id, instantiate_function


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
        # Handle multiple global minima (2D array or tuple of tuples) - use first one
        if isinstance(x_global, np.ndarray) and x_global.ndim == 2:
            x_global = x_global[0]
        elif isinstance(x_global, tuple) and len(x_global) > 0 and isinstance(x_global[0], tuple):
            x_global = x_global[0]

        if isinstance(x_global, (list, tuple, np.ndarray)):
            params = {f"x{i}": float(v) for i, v in enumerate(x_global)}
        else:
            params = x_global

        result = func(params)
        np.testing.assert_almost_equal(result, f_global, decimal=3)


@pytest.mark.bbob
class TestBBOBGlobalOptima:
    """Test global optima for BBOB functions."""

    @pytest.mark.parametrize("func_class", bbob_functions, ids=func_id)
    def test_bbob_has_global_minimum(self, func_class):
        """BBOB functions have known global minimum."""
        func = instantiate_function(func_class, n_dim=2)
        assert hasattr(func, "f_global") or "f_global" in func.spec
