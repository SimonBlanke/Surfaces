import pytest

from hyperactive import Hyperactive
from surfaces import mathematical_functions, machine_learning_functions


mathematical_functions_d = (
    "test_function",
    mathematical_functions,
)


machine_learning_functions_d = (
    "test_function",
    machine_learning_functions,
)


@pytest.mark.parametrize(*mathematical_functions_d)
def test_all_mathematical_functions(test_function):
    try:
        test_function_ = test_function()
    except TypeError:
        test_function_ = test_function(n_dim=2)

    hyper = Hyperactive()
    hyper.add_search(
        test_function_,
        test_function_.create_n_dim_search_space(value_types="list"),
        n_iter=15,
    )
    hyper.run()


@pytest.mark.parametrize(*machine_learning_functions_d)
def test_all_machine_learning_functions(test_function):
    test_function_ = test_function()

    hyper = Hyperactive()
    hyper.add_search(
        test_function_,
        test_function_.search_space,
        n_iter=15,
    )
    hyper.run()
