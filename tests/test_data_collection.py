import os
import pytest

from surfaces.test_functions import mathematical_functions, machine_learning_functions
from surfaces.test_functions.machine_learning import KNeighborsRegressorFunction
from surfaces.data_collector import SurfacesDataCollector

here_path = os.path.dirname(os.path.realpath(__file__))
search_data_path = os.path.join(here_path, "search_data_db")


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
        test_function_ = test_function(n_dim=3)

    objective_function = test_function_.objective_function
    search_space = test_function_.search_space(value_types="array", size=100)

    sdc = SurfacesDataCollector(path=search_data_path)
    sdc.collect(objective_function, search_space)
    sdc.remove()


def test_all_machine_learning_functions():
    test_function_ = KNeighborsRegressorFunction()

    objective_function = test_function_.objective_function
    search_space = test_function_.search_space(
        n_neighbors=[3, 4, 5], cv=[2], dataset=[test_function_.dataset_default[0]]
    )

    sdc = SurfacesDataCollector(path=search_data_path)
    sdc.collect(objective_function, search_space)
    sdc.remove()
