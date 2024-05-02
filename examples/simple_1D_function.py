from surfaces.test_functions.mathematical import GramacyAndLeeFunction


styblinski_tang_function = GramacyAndLeeFunction()

objective_function = styblinski_tang_function.objective_function
search_space = styblinski_tang_function.search_space()
