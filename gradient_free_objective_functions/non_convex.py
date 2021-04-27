import numpy as np


def ackley_function(A=20):
    def _ackley_function_(params):
        x1 = params["x0"]
        x2 = params["x1"]

        loss1 = -A * np.exp(-0.2 * np.sqrt(0.5 * (x1 * x1 + x2 * x2)))
        loss2 = -np.exp(0.5 * (np.cos(x1) + np.cos(x2)))
        loss3 = np.exp(1)
        loss4 = A

        score = -(loss1 + loss2 + loss3 + loss4)
        return score

    _ackley_function_.__name__ = ackley_function.__name__
    return _ackley_function_


def rastrigin_function(n_dim, A=1, B=1):
    def _rastrigin_function_(params):
        loss = 0
        for dim in range(n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += A * n_dim + (x * x - A * np.cos(B * x))

        score = -loss
        return score

    _rastrigin_function_.__name__ = rastrigin_function.__name__
    return _rastrigin_function_
