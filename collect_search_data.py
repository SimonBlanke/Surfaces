from surfaces.test_functions import machine_learning_functions
from surfaces.data_collector import SurfacesDataCollector


for machine_learning_function in machine_learning_functions:
    test_function = machine_learning_function(evaluate_from_data=False)

    objective_function = test_function.objective_function
    search_space = test_function.search_space()

    sdc = SurfacesDataCollector()
    sdc.collect(objective_function, search_space)
