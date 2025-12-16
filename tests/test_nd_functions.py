import pytest

from surfaces.test_functions import mathematical_functions


mathematical_functions_d = (
    "test_function",
    mathematical_functions,
)


@pytest.mark.parametrize(*mathematical_functions_d)
def test_all_mathematical_functions(test_function):
    try:
        test_function_ = test_function()
    except TypeError:
        test_function_ = test_function(n_dim=2)

    # Test that the function can be created and evaluated
    search_space = test_function_.create_n_dim_search_space(value_types="list")
    assert isinstance(search_space, dict)
    assert len(search_space) > 0
    
    # Test that the objective function works
    sample_params = {key: values[0] for key, values in search_space.items()}
    result = test_function_(sample_params)
    assert isinstance(result, (int, float))
    # Note: Some functions may return NaN for certain parameter combinations, which is valid
