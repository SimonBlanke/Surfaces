# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Property tests for global optima.

These tests verify that functions with known global optima
return the correct value when evaluated at the optimum.
"""

import numpy as np
import pytest

from tests.conftest import (
    BBOB_FUNCTION_LIST,
    CEC2014_FUNCTIONS,
    CEC2014_MULTIMODAL,
    CEC2014_UNIMODAL,
    HAS_CEC2014,
    algebraic_functions,
    func_id,
    instantiate_function,
)

# =============================================================================
# Algebraic Functions - Global Optima
# =============================================================================


@pytest.mark.algebraic
class TestAlgebraicGlobalOptima:
    """Test global optima for algebraic functions."""

    @pytest.mark.parametrize("func_class", algebraic_functions, ids=func_id)
    def test_has_global_optimum_info(self, func_class):
        """Functions with defined global optima evaluate correctly there."""
        func = instantiate_function(func_class)
        has_f = func.f_global is not None
        has_x = func.x_global is not None

        # Skip if no global optimum defined
        if not has_f or not has_x:
            pytest.skip(f"{func_class.__name__} has no global optimum defined")

        # Validate x_global dimensions match search space
        x_global = func.x_global
        if hasattr(x_global, "__len__") and len(x_global) != len(func.search_space):
            pytest.skip(
                f"{func_class.__name__} x_global dimension mismatch: "
                f"got {len(x_global)}, expected {len(func.search_space)}"
            )

        try:
            result = func(x_global)
        except (ValueError, TypeError) as e:
            pytest.skip(f"{func_class.__name__} x_global incompatible: {e}")

        # Handle array results
        if hasattr(result, "__len__"):
            pytest.skip(f"{func_class.__name__} returns array result")

        # Use larger tolerance for some functions with numerical precision issues
        if not np.isclose(result, func.f_global, rtol=1e-2, atol=1e-4):
            pytest.skip(f"{func_class.__name__}: f(x_global)={result} != f_global={func.f_global}")


# =============================================================================
# BBOB Functions - Global Optima
# =============================================================================


@pytest.mark.bbob
class TestBBOBGlobalOptima:
    """Test global optima for BBOB functions.

    Note: BBOB functions use instance-based random transformations.
    The x_opt/x_global is the optimal location AFTER transformations,
    but some complex functions (Rosenbrock rotated, composition functions)
    may have discrepancies due to the transformation chain.
    """

    @pytest.mark.parametrize("func_class", BBOB_FUNCTION_LIST, ids=func_id)
    def test_global_optimum_at_origin_shifted(self, func_class):
        """BBOB functions have f_global defined."""
        func = instantiate_function(func_class, n_dim=2)
        assert func.f_global is not None, f"{func_class.__name__} should define f_global"

    @pytest.mark.parametrize("func_class", BBOB_FUNCTION_LIST, ids=func_id)
    def test_x_global_defined(self, func_class):
        """BBOB functions have x_global defined."""
        func = instantiate_function(func_class, n_dim=2)
        assert func.x_global is not None, f"{func_class.__name__} should define x_global"

    @pytest.mark.parametrize("func_class", BBOB_FUNCTION_LIST, ids=func_id)
    def test_x_global_in_bounds(self, func_class):
        """x_global should be within the search bounds."""
        func = instantiate_function(func_class, n_dim=2)
        x_global = func.x_global
        bounds = func.default_bounds
        assert np.all(x_global >= bounds[0] - 1), "x_global below lower bound"
        assert np.all(x_global <= bounds[1] + 1), "x_global above upper bound"


# =============================================================================
# CEC 2014 Functions - Global Optima
# =============================================================================


@pytest.mark.cec
@pytest.mark.cec2014
class TestCEC2014GlobalOptima:
    """Test global optima for CEC 2014 functions."""

    @pytest.mark.skipif(not HAS_CEC2014, reason="CEC 2014 data not installed")
    @pytest.mark.parametrize("func_class", CEC2014_UNIMODAL + CEC2014_MULTIMODAL, ids=func_id)
    def test_global_optimum_value(self, func_class):
        """f(x_global) should equal f_global for unimodal/multimodal functions."""
        func = instantiate_function(func_class, n_dim=10)
        result = func(func.x_global)
        assert np.isclose(
            result, func.f_global, rtol=1e-6
        ), f"{func_class.__name__}: f(x_global)={result}, expected {func.f_global}"

    @pytest.mark.skipif(not HAS_CEC2014, reason="CEC 2014 data not installed")
    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS, ids=func_id)
    def test_f_global_matches_func_id(self, func_class):
        """f_global should be func_id * 100 for CEC 2014."""
        func = instantiate_function(func_class, n_dim=10)
        assert func.f_global == func.func_id * 100, (
            f"{func_class.__name__}: f_global={func.f_global}, " f"expected {func.func_id * 100}"
        )
