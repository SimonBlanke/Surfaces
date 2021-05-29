import pytest
import numpy as np
from gradient_free_optimizers import RandomSearchOptimizer

from bbox_functions.convex import (
    SphereFunction,
)
from bbox_functions.non_convex import (
    RastriginFunction,
    AckleyFunction,
    RosenbrockFunction,
    BealeFunction,
    HimmelblausFunction,
    HölderTableFunction,
    CrossInTrayFunction,
    SimionescuFunction,
)

sphere_function = SphereFunction(2, input_type="arrays")
rastrigin_function = RastriginFunction(2, input_type="arrays")
ackley_function = AckleyFunction(input_type="arrays")
rosenbrock_function = RosenbrockFunction(input_type="arrays")
beale_function = BealeFunction(input_type="arrays")
himmelblaus_function = HimmelblausFunction(input_type="arrays")
hölder_table_function = HölderTableFunction(input_type="arrays")
cross_in_tray_function = CrossInTrayFunction(input_type="arrays")
simionescu_function = SimionescuFunction(input_type="arrays")


objective_function_para_2D = (
    "objective_function",
    [
        (sphere_function),
        (rastrigin_function),
        (ackley_function),
        (rosenbrock_function),
        (beale_function),
        (himmelblaus_function),
        (hölder_table_function),
        (cross_in_tray_function),
        (simionescu_function),
    ],
)

a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 2, 3, 4, 5])


@pytest.mark.parametrize(*objective_function_para_2D)
def test_array_input(objective_function):
    objective_function(a, b)