import os
import numpy as np

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
)
from bbox_functions.visualize import (
    matplotlib_heatmap,
    matplotlib_surface,
)

path = os.path.realpath(__file__).rsplit("/", 1)[0]


sphere_function = SphereFunction(2, metric="loss")
rastrigin_function = RastriginFunction(2, metric="loss")
ackley_function = AckleyFunction(metric="loss")
rosenbrock_function = RosenbrockFunction(metric="loss")
beale_function = BealeFunction(metric="loss")
himmelblaus_function = HimmelblausFunction(metric="loss")
hölder_table_function = HölderTableFunction(metric="loss")
cross_in_tray_function = CrossInTrayFunction(metric="score")


resolution = 0.05

search_space1 = {
    "x0": np.arange(-5, 5, resolution),
    "x1": np.arange(-5, 5, resolution),
}

search_space2 = {
    "x0": np.arange(-10, 10, resolution),
    "x1": np.arange(-10, 10, resolution),
}

search_space2 = {
    "x0": np.arange(-7, 7, resolution),
    "x1": np.arange(-7, 7, resolution),
}


objective_function_infos = {
    sphere_function: {
        "search_space": search_space1,
        "norm": None,
    },
    rastrigin_function: {
        "search_space": search_space1,
        "norm": None,
    },
    ackley_function: {
        "search_space": search_space1,
        "norm": None,
    },
    rosenbrock_function: {
        "search_space": search_space1,
        "norm": None,
    },
    beale_function: {
        "search_space": search_space1,
        "norm": "color_log",
    },
    himmelblaus_function: {
        "search_space": search_space1,
        "norm": "color_log",
    },
    hölder_table_function: {
        "search_space": search_space2,
        "norm": None,
    },
    cross_in_tray_function: {
        "search_space": search_space2,
        "norm": None,
    },
}


for objective_function in objective_function_infos.keys():
    objective_function_info = objective_function_infos[objective_function]

    name = objective_function.__name__
    search_space = objective_function_info["search_space"]
    norm = objective_function_info["norm"]

    print(name, "\n")

    matplotlib_heatmap(objective_function, search_space, norm=norm).savefig(
        path + "/images/" + name + "_heatmap.jpg", dpi=100
    )
    matplotlib_surface(objective_function, search_space, norm=norm).savefig(
        path + "/images/" + name + "_surface.jpg", dpi=100
    )
