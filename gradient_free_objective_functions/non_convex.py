# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .base_objective_function import ObjectiveFunction


class AckleyFunction(ObjectiveFunction):
    def __init__(self, A=20, B=2 * np.pi, metric="score"):
        super().__init__()
        self.__name__ = "ackley_function"

        self.A = A
        self.B = B

    def _objective_function(self, params):
        x1 = params["x0"]
        x2 = params["x1"]

        loss1 = -self.A * np.exp(-0.2 * np.sqrt(0.5 * (x1 * x1 + x2 * x2)))
        loss2 = -np.exp(0.5 * (np.cos(self.B * x1) + np.cos(self.B * x2)))
        loss3 = np.exp(1)
        loss4 = self.A

        loss = loss1 + loss2 + loss3 + loss4

        return self.return_metric(loss)

    def __call__(self, params):
        return self._objective_function(params)


class RastriginFunction(ObjectiveFunction):
    def __init__(self, n_dim, A=1, B=2 * np.pi, metric="score"):
        super().__init__()
        self.__name__ = "rastrigin_function"

        self.n_dim = n_dim
        self.A = A
        self.B = B

    def _objective_function(self, params):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += self.A * self.n_dim + (x * x - self.A * np.cos(self.B * x))

        return self.return_metric(loss)

    def __call__(self, params):
        return self._objective_function(params)
