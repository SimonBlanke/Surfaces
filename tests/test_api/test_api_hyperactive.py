import pytest

from hyperactive import Hyperactive
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

    objective_function = test_function_.objective_function
    n_iter = 5

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=n_iter,
    )
    hyper.run()
