"""Integration tests for Gradient-Free-Optimizers.

Tests that Surfaces functions work correctly with the GFO library.
"""

import numpy as np
import pytest
from gradient_free_optimizers import RandomSearchOptimizer

from surfaces.test_functions.algebraic import (
    AckleyFunction,
    BealeFunction,
    BoothFunction,
    BukinFunctionN6,
    CrossInTrayFunction,
    EasomFunction,
    GoldsteinPriceFunction,
    HimmelblausFunction,
    HölderTableFunction,
    RastriginFunction,
    RosenbrockFunction,
    SimionescuFunction,
    SphereFunction,
    StyblinskiTangFunction,
)

pytestmark = pytest.mark.slow


# 2D test functions
FUNCTIONS_2D = [
    SphereFunction(2),
    RastriginFunction(2),
    AckleyFunction(),
    RosenbrockFunction(2),
    BealeFunction(),
    HimmelblausFunction(),
    HölderTableFunction(),
    CrossInTrayFunction(),
    SimionescuFunction(),
    EasomFunction(),
    BoothFunction(),
    GoldsteinPriceFunction(),
    StyblinskiTangFunction(2),
    BukinFunctionN6(),
]


@pytest.mark.parametrize("test_function", FUNCTIONS_2D, ids=lambda f: f.__class__.__name__)
def test_optimization_2d(test_function):
    """Test GFO optimization with 2D functions."""
    search_space = {
        "x0": np.arange(0, 100, 1),
        "x1": np.arange(0, 100, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(test_function, n_iter=30)


# 3D test functions
FUNCTIONS_3D = [
    SphereFunction(3),
    RastriginFunction(3),
    RosenbrockFunction(3),
]


@pytest.mark.parametrize("test_function", FUNCTIONS_3D, ids=lambda f: f.__class__.__name__)
def test_optimization_3d(test_function):
    """Test GFO optimization with 3D functions."""
    search_space = {
        "x0": np.arange(0, 100, 1),
        "x1": np.arange(0, 100, 1),
        "x2": np.arange(0, 100, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(test_function, n_iter=30)


# 4D test functions
FUNCTIONS_4D = [
    SphereFunction(4),
    RastriginFunction(4),
    RosenbrockFunction(4),
]


@pytest.mark.parametrize("test_function", FUNCTIONS_4D, ids=lambda f: f.__class__.__name__)
def test_optimization_4d(test_function):
    """Test GFO optimization with 4D functions."""
    search_space = {
        "x0": np.arange(0, 100, 1),
        "x1": np.arange(0, 100, 1),
        "x2": np.arange(0, 100, 1),
        "x3": np.arange(0, 100, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(test_function, n_iter=30)
