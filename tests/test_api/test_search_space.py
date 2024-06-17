import pytest
from gradient_free_optimizers import RandomSearchOptimizer
from hyperactive import Hyperactive


from surfaces.test_functions import (
    machine_learning_functions,
)
from surfaces.test_functions.mathematical import (
    mathematical_functions_1d,
    mathematical_functions_2d,
    mathematical_functions_nd,
)

mathematical_functions_1d = (
    "test_function",
    mathematical_functions_1d,
)

mathematical_functions_2d = (
    "test_function",
    mathematical_functions_2d,
)

mathematical_functions_nd = (
    "test_function",
    mathematical_functions_nd,
)


machine_learning_functions_d = (
    "test_function",
    machine_learning_functions,
)


@pytest.mark.parametrize(*mathematical_functions_1d)
def test_search_space_1d_0(test_function):
    test_function_ = test_function()

    objective_function = test_function_.objective_function
    search_space = test_function_.search_space(min=-1, max=1, value_types="array")
    n_iter = 20

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=n_iter)


@pytest.mark.parametrize(*mathematical_functions_2d)
def test_search_space_2d_0(test_function):
    print("\n test_function", test_function, "\n")
    test_function_ = test_function()

    objective_function = test_function_.objective_function
    search_space = test_function_.search_space(min=-1, max=1, value_types="array")
    n_iter = 20

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=n_iter)


@pytest.mark.parametrize(*mathematical_functions_2d)
def test_search_space_2d_1(test_function):
    test_function_ = test_function()

    objective_function = test_function_.objective_function
    search_space = test_function_.search_space(
        min=[-1, -1], max=[1, 1], value_types="array"
    )
    n_iter = 20

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=n_iter)


@pytest.mark.parametrize(*mathematical_functions_nd)
def test_search_space_nd_0(test_function):
    test_function_ = test_function(n_dim=3)

    objective_function = test_function_.objective_function
    search_space = test_function_.search_space(min=-1, max=1, value_types="array")
    n_iter = 20

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=n_iter)


@pytest.mark.parametrize(*mathematical_functions_nd)
def test_search_space_nd_1(test_function):
    test_function_ = test_function(n_dim=3)

    objective_function = test_function_.objective_function
    search_space = test_function_.search_space(
        min=[-1, -1, -1], max=[1, 1, 1], value_types="array"
    )
    n_iter = 20

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=n_iter)


@pytest.mark.parametrize(*machine_learning_functions_d)
def test_all_(test_function):
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
