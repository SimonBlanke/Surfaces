import numpy as np


def ackley_function(params):
    x = params["x1"]
    y = params["x2"]

    A = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
    B = -np.exp(0.5 * (np.cos(x) + np.cos(y)))
    C = np.exp(1)
    D = 20

    score = -(A + B + C + D)

    return score
