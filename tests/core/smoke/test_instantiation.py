"""Smoke tests for core function instantiation.

These tests verify that algebraic, engineering, and BBOB functions
can be instantiated and called without errors. They require only numpy.
"""

import numpy as np
import pytest

from surfaces.test_functions.algebraic import (
    algebraic_functions_1d,
    algebraic_functions_2d,
    algebraic_functions_nd,
)
from surfaces.test_functions.bbob import BBOB_FUNCTIONS
from surfaces.test_functions.engineering import engineering_functions

BBOB_FUNCTION_LIST = list(BBOB_FUNCTIONS.values())


def func_id(func_class):
    """Generate readable test IDs from function classes."""
    return func_class.__name__


def instantiate_function(func_class, **kwargs):
    """Instantiate a function class with appropriate parameters.

    Uses the class's _spec['scalable'] attribute to determine if n_dim is required.
    """
    spec = getattr(func_class, "_spec", {})
    is_scalable = spec.get("scalable", False)

    if is_scalable and "n_dim" not in kwargs:
        kwargs["n_dim"] = 2
    return func_class(**kwargs)


def get_sample_params(func):
    """Get sample parameters from a function's search space."""
    params = {}
    for key, values in func.search_space.items():
        if hasattr(values, "__iter__") and not isinstance(values, str):
            values_list = list(values)
            params[key] = values_list[len(values_list) // 2]
        else:
            params[key] = values
    return params


# =============================================================================
# Algebraic Functions
# =============================================================================


@pytest.mark.smoke
@pytest.mark.algebraic
class TestAlgebraicInstantiation:
    """Smoke tests for algebraic function instantiation."""

    @pytest.mark.parametrize("func_class", algebraic_functions_1d, ids=func_id)
    def test_1d_functions(self, func_class):
        """1D functions instantiate and evaluate correctly."""
        func = instantiate_function(func_class)
        assert len(func.search_space) == 1
        result = func(get_sample_params(func))
        assert isinstance(result, (int, float))

    @pytest.mark.parametrize("func_class", algebraic_functions_2d, ids=func_id)
    def test_2d_functions(self, func_class):
        """2D functions instantiate and evaluate correctly."""
        func = instantiate_function(func_class)
        assert len(func.search_space) == 2
        result = func(get_sample_params(func))
        assert isinstance(result, (int, float))

    @pytest.mark.parametrize("func_class", algebraic_functions_nd, ids=func_id)
    @pytest.mark.parametrize("n_dim", [2, 5, 10])
    def test_nd_functions(self, func_class, n_dim):
        """N-D functions instantiate with various dimensions."""
        func = instantiate_function(func_class, n_dim=n_dim)
        assert len(func.search_space) == n_dim
        result = func(get_sample_params(func))
        assert isinstance(result, (int, float))


# =============================================================================
# Engineering Functions
# =============================================================================


@pytest.mark.smoke
@pytest.mark.engineering
class TestEngineeringInstantiation:
    """Smoke tests for engineering function instantiation."""

    @pytest.mark.parametrize("func_class", engineering_functions, ids=func_id)
    def test_engineering_functions(self, func_class):
        """Engineering functions instantiate and evaluate correctly."""
        func = instantiate_function(func_class)
        assert len(func.search_space) > 0
        result = func(get_sample_params(func))
        assert isinstance(result, (int, float))

    @pytest.mark.parametrize("func_class", engineering_functions, ids=func_id)
    def test_engineering_has_constraints(self, func_class):
        """Engineering functions have constraint methods."""
        func = instantiate_function(func_class)
        assert hasattr(func, "constraints")
        assert hasattr(func, "is_feasible")
        assert hasattr(func, "raw_objective")


# =============================================================================
# BBOB Functions
# =============================================================================


@pytest.mark.smoke
@pytest.mark.bbob
class TestBBOBInstantiation:
    """Smoke tests for BBOB function instantiation."""

    @pytest.mark.parametrize("func_class", BBOB_FUNCTION_LIST, ids=func_id)
    def test_bbob_functions(self, func_class):
        """BBOB functions instantiate and evaluate correctly."""
        func = instantiate_function(func_class, n_dim=2)
        assert len(func.search_space) == 2
        result = func(get_sample_params(func))
        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", BBOB_FUNCTION_LIST, ids=func_id)
    def test_bbob_has_func_id(self, func_class):
        """BBOB functions have func_id in spec."""
        func = instantiate_function(func_class, n_dim=2)
        assert func.spec.get("func_id") is not None
        assert 1 <= func.spec["func_id"] <= 24
