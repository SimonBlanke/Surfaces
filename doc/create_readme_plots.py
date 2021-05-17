import os
import numpy as np

from gradient_free_objective_functions.convex import (
    SphereFunction,
)
from gradient_free_objective_functions.non_convex import (
    RastriginFunction,
    AckleyFunction,
    RosenbrockFunction,
    BealeFunction,
)
from gradient_free_objective_functions.visualize import (
    matplotlib_heatmap,
    matplotlib_surface,
)

path = os.path.realpath(__file__).rsplit("/", 1)[0]


sphere_function = SphereFunction(2, metric="loss")
rastrigin_function = RastriginFunction(2, metric="loss")
ackley_function = AckleyFunction(metric="loss")
rosenbrock_function = RosenbrockFunction(metric="loss")
beale_function = BealeFunction(metric="loss")


objective_functions = [
    sphere_function,
    rastrigin_function,
    ackley_function,
    rosenbrock_function,
    beale_function,
]

search_space = {
    "x0": np.arange(-5, 5, 0.01),
    "x1": np.arange(-5, 5, 0.01),
}


objective_function_infos = {
    sphere_function: {
        "name": "sphere_function",
        "search_space": search_space,
        "norm": None,
    },
    rastrigin_function: {
        "name": "rastrigin_function",
        "search_space": search_space,
        "norm": None,
    },
    ackley_function: {
        "name": "ackley_function",
        "search_space": search_space,
        "norm": None,
    },
    rosenbrock_function: {
        "name": "rosenbrock_function",
        "search_space": search_space,
        "norm": None,
    },
    beale_function: {
        "name": "beale_function",
        "search_space": search_space,
        "norm": "color_log",
    },
}


for objective_function in objective_functions:
    objective_function_info = objective_function_infos[objective_function]

    name = objective_function_info["name"]
    search_space = objective_function_info["search_space"]
    norm = objective_function_info["norm"]

    matplotlib_heatmap(objective_function, search_space, norm=norm).savefig(
        path + "/images/" + name + "_heatmap.jpg", dpi=100
    )

    matplotlib_surface(objective_function, search_space, norm=norm).savefig(
        path + "/images/" + name + "_surface.jpg", dpi=100
    )
