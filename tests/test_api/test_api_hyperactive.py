import pytest

from surfaces.test_functions import test_functions


test_functions_d = (
    "test_function",
    test_functions,
)


@pytest.mark.parametrize(*test_functions_d)
def test_all_functions(test_function):
    try:
        test_function_ = test_function()
    except TypeError:
        test_function_ = test_function(n_dim=3)

    try:
        search_space = test_function_.search_space(value_types="list")
    except TypeError:
        search_space = test_function_.search_space()

    # Test that the function can be created and evaluated
    assert isinstance(search_space, dict)
    assert len(search_space) > 0
    
    # Test that the function works
    sample_params = {key: list(values)[0] if hasattr(values, '__iter__') else values
                    for key, values in search_space.items()}
    result = test_function_(sample_params)
    assert isinstance(result, (int, float))
