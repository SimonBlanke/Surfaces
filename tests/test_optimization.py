import pytest
import numpy as np
from gradient_free_optimizers import RandomSearchOptimizer

from gradient_free_objective_functions.convex import (
    SphereFunction,
)
from gradient_free_objective_functions.non_convex import (
    RastriginFunction,
    AckleyFunction,
    RosenbrockFunction,
    BealeFunction,
)

sphere_function = SphereFunction(2)
rastrigin_function = RastriginFunction(2)
ackley_function = AckleyFunction()
rosenbrock_function = RosenbrockFunction()
beale_function = BealeFunction()


objective_function_para_2D = (
    "objective_function",
    [
        (sphere_function),
        (rastrigin_function),
        (ackley_function),
        (rosenbrock_function),
        (beale_function),
    ],
)


@pytest.mark.parametrize(*objective_function_para_2D)
def test_optimization_2D(objective_function):
    search_space = {
        "x0": np.arange(0, 100, 1),
        "x1": np.arange(0, 100, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=30)


############################################################

sphere_function = SphereFunction(3)
rastrigin_function = RastriginFunction(3)


objective_function_para_3D = (
    "objective_function",
    [
        (sphere_function),
        (rastrigin_function),
    ],
)


@pytest.mark.parametrize(*objective_function_para_3D)
def test_optimization_3D(objective_function):
    search_space = {
        "x0": np.arange(0, 100, 1),
        "x1": np.arange(0, 100, 1),
        "x2": np.arange(0, 100, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=30)


############################################################

sphere_function = SphereFunction(4)
rastrigin_function = RastriginFunction(4)


objective_function_para_4D = (
    "objective_function",
    [
        (sphere_function),
        (rastrigin_function),
    ],
)


@pytest.mark.parametrize(*objective_function_para_4D)
def test_optimization_4D(objective_function):
    search_space = {
        "x0": np.arange(0, 100, 1),
        "x1": np.arange(0, 100, 1),
        "x2": np.arange(0, 100, 1),
        "x3": np.arange(0, 100, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=30)