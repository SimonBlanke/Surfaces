from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction


k_neighbors_classifier = KNeighborsClassifierFunction()

objective_function = k_neighbors_classifier.objective_function
search_space = k_neighbors_classifier.search_space()
