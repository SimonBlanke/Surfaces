from surfaces.test_functions import mathematical_functions, machine_learning_functions
from surfaces.data_collector import SurfacesDataCollector


for mathematical_function in mathematical_functions:
    print("\n ---", mathematical_function.name, "---")
    try:
        test_function = mathematical_function(metric="loss")
    except TypeError:
        test_function = mathematical_function(n_dim=2, metric="loss")

    objective_function = test_function.objective_function
    search_space = test_function.search_space(value_types="array")

    sdc = SurfacesDataCollector()
    sdc.collect(objective_function, search_space)


for machine_learning_function in machine_learning_functions:
    test_function = machine_learning_function(load_search_data=False)

    objective_function = test_function.objective_function
    search_space = test_function.search_space(value_types="array")

    sdc = SurfacesDataCollector()
    sdc.collect(objective_function, search_space)
