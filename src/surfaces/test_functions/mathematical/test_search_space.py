import pytest

from surfaces import (
    mathematical_functions_1d,
    mathematical_functions_2d,
    mathematical_functions_nd,
    machine_learning_functions,
)


mathematical_functions_1d = (
    "test_function",
    mathematical_functions_1d,
)

mathematical_functions_2d = (
    "test_function",
    mathematical_functions_1d,
)

mathematical_functions_nd = (
    "test_function",
    mathematical_functions_1d,
)


machine_learning_functions_d = (
    "test_function",
    machine_learning_functions,
)


@pytest.mark.parametrize(*mathematical_functions_d)
def test_(test_function):
    try:
        test_function_ = test_function()
    except TypeError:
        test_function_ = test_function(n_dim=2)

    objective_function = test_function_.objective_function
    search_space = test_function_.search_space(value_types="array")
    n_iter = 20


@pytest.mark.parametrize(*machine_learning_functions_d)
def test_all_(test_function):
    test_function_ = test_function()

    objective_function = test_function_.objective_function
    search_space = test_function_.search_space()
    n_iter = 20
