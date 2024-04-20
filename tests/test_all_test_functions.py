import pytest

from hyperactive import Hyperactive
from gradient_free_optimizers import RandomSearchOptimizer

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
def test_all_mathematical_functions(test_function):
    try:
        test_function_ = test_function()
    except TypeError:
        test_function_ = test_function(n_dim=2)

    objective_function = test_function_.objective_function
    search_space = test_function_.search_space(value_types="array")
    n_iter = 100

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=n_iter)


@pytest.mark.parametrize(*machine_learning_functions_d)
def test_all_machine_learning_functions(test_function):
    test_function_ = test_function()

    objective_function = test_function_.objective_function
    search_space = test_function_.search_space()
    n_iter = 3

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=n_iter,
    )
    hyper.run()
