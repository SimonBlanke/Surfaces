"""Smoke tests for full function instantiation.

These tests verify that CEC and ML functions can be instantiated
and called without errors. They require additional dependencies
like scikit-learn and CEC data packages.
"""

import inspect

import numpy as np
import pytest

import surfaces.test_functions.cec.cec2013 as cec2013
import surfaces.test_functions.cec.cec2014 as cec2014
import surfaces.test_functions.cec.cec2017 as cec2017
from surfaces.test_functions.machine_learning import machine_learning_functions

# Build CEC function lists dynamically
CEC2013_FUNCTIONS = [
    v
    for k, v in vars(cec2013).items()
    if inspect.isclass(v) and not k.startswith("_") and k != "CEC2013Function"
]

CEC2014_FUNCTIONS = [
    v
    for k, v in vars(cec2014).items()
    if inspect.isclass(v) and not k.startswith("_") and k != "CEC2014Function"
]

CEC2017_FUNCTIONS = [
    v
    for k, v in vars(cec2017).items()
    if inspect.isclass(v) and not k.startswith("_") and k != "CEC2017Function"
]


def func_id(func_class):
    """Generate readable test IDs from function classes."""
    return func_class.__name__


def instantiate_function(func_class, **kwargs):
    """Instantiate a function class with appropriate parameters."""
    try:
        return func_class(**kwargs)
    except TypeError:
        return func_class()


# =============================================================================
# CEC Functions
# =============================================================================


@pytest.mark.smoke
@pytest.mark.cec
@pytest.mark.cec2013
class TestCEC2013Instantiation:
    """Smoke tests for CEC 2013 function instantiation."""

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

    @pytest.mark.parametrize("func_class", machine_learning_functions, ids=func_id)
    def test_ml_functions_instantiate(self, func_class):
        """ML functions instantiate correctly."""
        func = instantiate_function(func_class)
        assert len(func.search_space) > 0
        assert callable(func)

    def test_ml_function_evaluates(self):
        """At least one ML function evaluates correctly (quick test)."""
        from surfaces.test_functions import KNeighborsClassifierFunction
        from surfaces.test_functions.machine_learning.tabular.classification.datasets import (
            iris_data,
        )

        func = KNeighborsClassifierFunction()
        params = {
            "n_neighbors": 5,
            "algorithm": "auto",
            "cv": 2,
            "dataset": iris_data,
        }
        result = func(params)
        assert isinstance(result, (int, float))
