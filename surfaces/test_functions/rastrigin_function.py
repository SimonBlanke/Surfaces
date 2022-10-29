# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .base_objective_function import ObjectiveFunction


class RastriginFunction(ObjectiveFunction):
    name = "Rastrigin Function"
    _name_ = "rastrigin_function"
    __name__ = "RastriginFunction"

    def __init__(
        self,
        n_dim,
        A=10,
        angle=2 * np.pi,
        metric="score",
        input_type="dictionary",
        sleep=0,
    ):
        super().__init__(metric, input_type, sleep)

        self.n_dim = n_dim
        self.A = A
        self.angle = angle

    def objective_function_dict(self, params):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += x * x - self.A * np.cos(self.angle * x)

        loss = self.A * self.n_dim + loss

        return self.return_metric(loss)
