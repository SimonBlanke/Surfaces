def sphere_function_1d(params):
    x = params["x1"]
    score = -(x * x)

    return score


def sphere_function_2d(params):
    x = params["x1"]
    y = params["x2"]
    score = -(x * x + y * y)

    return score


def sphere_function_3d(params):
    x = params["x1"]
    y = params["x2"]
    z = params["x3"]
    score = -(x * x + y * y + z * z)

    return score
