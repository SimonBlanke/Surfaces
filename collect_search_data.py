from surfaces import mathematical_functions, machine_learning_functions


for mathematical_function in mathematical_functions:
    print("\n ---", mathematical_function.name, "---")
    try:
        test_function = mathematical_function(metric="loss")
    except TypeError:
        test_function = mathematical_function(n_dim=2, metric="loss")

    test_function.collect_data()


for machine_learning_function in machine_learning_functions:
    test_function = machine_learning_function()
    test_function.collect_data()
