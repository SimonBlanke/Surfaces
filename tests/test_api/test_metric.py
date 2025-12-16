import pytest

from surfaces.test_functions import mathematical_functions, machine_learning_functions


mathematical_functions_d = (
    "test_function",
    mathematical_functions,
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

    search_space = test_function_.search_space(value_types="array")
    # Test function is directly callable via __call__
    assert callable(test_function_)


@pytest.mark.parametrize(*machine_learning_functions_d)
def test_all_(test_function):
    test_function_ = test_function()

    search_space = test_function_.search_space()
    # Test function is directly callable via __call__
    assert callable(test_function_)
