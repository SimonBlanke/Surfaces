# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .base_objective_function import ObjectiveFunction


class EasomFunction(ObjectiveFunction):
    name = "Easom Function"
    _name_ = "easom_function"
    __name__ = "EasomFunction"

    def __init__(
        self, A=-1, B=1, angle=1, metric="score", input_type="dictionary", sleep=0
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = self.A * np.cos(x * self.angle) * np.cos(y * self.angle)
        loss2 = np.exp(-((x - np.pi / self.B) ** 2 + (y - np.pi / self.B) ** 2))

        loss = loss1 * loss2

        return self.return_metric(loss)