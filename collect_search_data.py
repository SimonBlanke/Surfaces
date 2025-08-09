from surfaces.test_functions import machine_learning_functions
from surfaces.data_collector import SurfacesDataCollector


for machine_learning_function in machine_learning_functions:
    test_function = machine_learning_function(evaluate_from_data=False)

    sdc = SurfacesDataCollector()
    # Now uses default search space automatically and handles dataset objects
    sdc.collect_search_data(test_function)
