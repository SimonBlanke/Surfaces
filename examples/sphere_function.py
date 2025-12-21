from surfaces.test_functions.mathematical import SphereFunction

sphere_function_1d = SphereFunction(n_dim=1)

objective_function = sphere_function_1d.objective_function
search_space = sphere_function_1d.search_space()


sphere_function_2d = SphereFunction(n_dim=2)

objective_function = sphere_function_2d.objective_function
search_space = sphere_function_2d.search_space()


sphere_function_3d = SphereFunction(n_dim=3)

objective_function = sphere_function_3d.objective_function
search_space = sphere_function_3d.search_space()
