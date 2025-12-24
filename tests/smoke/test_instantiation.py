# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Smoke tests for function instantiation.

These tests verify that all test functions can be instantiated
and called without errors. They're designed to catch import
issues and basic API breakages quickly.
"""

import pytest
import numpy as np

from tests.conftest import (
    algebraic_functions,
    algebraic_functions_1d,
    algebraic_functions_2d,
    algebraic_functions_nd,
    engineering_functions,
    BBOB_FUNCTION_LIST,
    CEC2013_FUNCTIONS,
    CEC2014_FUNCTIONS,
    CEC2017_FUNCTIONS,
    machine_learning_functions,
    instantiate_function,
    get_sample_params,
    func_id,
    HAS_ML,
    HAS_CEC2013,
    HAS_CEC2014,
    HAS_CEC2017,
)


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
        assert hasattr(func, 'constraints')
        assert hasattr(func, 'is_feasible')
        assert hasattr(func, 'raw_objective')


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


# =============================================================================
# CEC Functions
# =============================================================================

@pytest.mark.smoke
@pytest.mark.cec
@pytest.mark.cec2013
class TestCEC2013Instantiation:
    """Smoke tests for CEC 2013 function instantiation."""

    @pytest.mark.skipif(not HAS_CEC2013, reason="CEC 2013 data not installed")
    @pytest.mark.parametrize("func_class", CEC2013_FUNCTIONS, ids=func_id)
    def test_cec2013_functions(self, func_class):
        """CEC 2013 functions instantiate and evaluate correctly."""
        func = instantiate_function(func_class, n_dim=10)
        assert len(func.search_space) == 10
        result = func(np.zeros(10))
        assert isinstance(result, (int, float))
        assert np.isfinite(result)


@pytest.mark.smoke
@pytest.mark.cec
@pytest.mark.cec2014
class TestCEC2014Instantiation:
    """Smoke tests for CEC 2014 function instantiation."""

    @pytest.mark.skipif(not HAS_CEC2014, reason="CEC 2014 data not installed")
    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS, ids=func_id)
    def test_cec2014_functions(self, func_class):
        """CEC 2014 functions instantiate and evaluate correctly."""
        func = instantiate_function(func_class, n_dim=10)
        assert len(func.search_space) == 10
        result = func(np.zeros(10))
        assert isinstance(result, (int, float))
        assert np.isfinite(result)


@pytest.mark.smoke
@pytest.mark.cec
@pytest.mark.cec2017
class TestCEC2017Instantiation:
    """Smoke tests for CEC 2017 function instantiation."""

    @pytest.mark.skipif(not HAS_CEC2017, reason="CEC 2017 data not installed")
    @pytest.mark.parametrize("func_class", CEC2017_FUNCTIONS, ids=func_id)
    def test_cec2017_functions(self, func_class):
        """CEC 2017 functions instantiate and evaluate correctly."""
        func = instantiate_function(func_class, n_dim=10)
        assert len(func.search_space) == 10
        result = func(np.zeros(10))
        assert isinstance(result, (int, float))
        assert np.isfinite(result)


# =============================================================================
# Machine Learning Functions
# =============================================================================

@pytest.mark.smoke
@pytest.mark.ml
class TestMLInstantiation:
    """Smoke tests for machine learning function instantiation."""

    @pytest.mark.skipif(not HAS_ML, reason="scikit-learn not installed")
    @pytest.mark.parametrize("func_class", machine_learning_functions, ids=func_id)
    def test_ml_functions_instantiate(self, func_class):
        """ML functions instantiate correctly."""
        func = instantiate_function(func_class)
        assert len(func.search_space) > 0
        assert callable(func)

    @pytest.mark.skipif(not HAS_ML, reason="scikit-learn not installed")
    def test_ml_function_evaluates(self, quick_ml_params):
        """At least one ML function evaluates correctly (quick test)."""
        from surfaces.test_functions import KNeighborsClassifierFunction

        func = KNeighborsClassifierFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)
        assert isinstance(result, (int, float))
