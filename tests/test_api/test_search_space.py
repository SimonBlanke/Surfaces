import pytest
from gradient_free_optimizers import RandomSearchOptimizer

from surfaces.test_functions import (
    machine_learning_functions,
)
from surfaces.test_functions.algebraic import (
    algebraic_functions_1d,
    algebraic_functions_2d,
    algebraic_functions_nd,
)

algebraic_functions_1d_param = (
    "test_function",
    algebraic_functions_1d,
)

algebraic_functions_2d_param = (
    "test_function",
    algebraic_functions_2d,
)

algebraic_functions_nd_param = (
    "test_function",
    algebraic_functions_nd,
)


machine_learning_functions_d = (
    "test_function",
    machine_learning_functions,
)


@pytest.mark.parametrize(*algebraic_functions_1d_param)
def test_search_space_1d_default(test_function):
    """Test that 1D functions have valid search_space."""
    test_function_ = test_function()

    search_space = test_function_.search_space
    n_iter = 20

    opt = RandomSearchOptimizer(search_space)
    opt.search(test_function_, n_iter=n_iter)


@pytest.mark.parametrize(*algebraic_functions_2d_param)
def test_search_space_2d_default(test_function):
    """Test that 2D functions have valid search_space."""
    test_function_ = test_function()

    search_space = test_function_.search_space
    n_iter = 20

    opt = RandomSearchOptimizer(search_space)
    opt.search(test_function_, n_iter=n_iter)


@pytest.mark.parametrize(*algebraic_functions_nd_param)
def test_search_space_nd_default(test_function):
    """Test that ND functions have valid search_space."""
    test_function_ = test_function(n_dim=3)

    search_space = test_function_.search_space
    n_iter = 20

    opt = RandomSearchOptimizer(search_space)
    opt.search(test_function_, n_iter=n_iter)


@pytest.mark.parametrize(*algebraic_functions_nd_param)
def test_search_space_nd_custom_bounds(test_function):
    """Test internal _search_space with custom bounds."""
    test_function_ = test_function(n_dim=3)

    # Use internal method for custom bounds
    search_space = test_function_._search_space(
        min=[-1, -1, -1], max=[1, 1, 1], value_types="array"
    )
    n_iter = 20

    opt = RandomSearchOptimizer(search_space)
    opt.search(test_function_, n_iter=n_iter)


@pytest.mark.slow
@pytest.mark.ml
@pytest.mark.parametrize(*machine_learning_functions_d)
def test_ml_search_space(test_function):
    """Test that ML functions have valid search_space."""
    try:
        test_function_ = test_function()
    except ImportError as e:
        pytest.skip(f"Optional dependency not installed: {e}")

    search_space = test_function_.search_space
    assert isinstance(search_space, dict)
    assert len(search_space) > 0

    # Test that the function works
    sample_params = {
        key: list(values)[0] if hasattr(values, "__iter__") else values
        for key, values in search_space.items()
    }
    result = test_function_(sample_params)
    assert isinstance(result, (int, float))
