# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import ObjectiveFunction


class AckleyFunction(ObjectiveFunction):
    name = "Ackley Function"
    _name_ = "ackley_function"
    __name__ = "AckleyFunction"

    def __init__(
        self, A=20, angle=2 * np.pi, metric="score", input_type="dictionary", sleep=0
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = -self.A * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
        loss2 = -np.exp(0.5 * (np.cos(self.angle * x) + np.cos(self.angle * y)))
        loss3 = np.exp(1)
        loss4 = self.A

        loss = loss1 + loss2 + loss3 + loss4

        return self.return_metric(loss)
