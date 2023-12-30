# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import MathematicalFunction


class SimionescuFunction(MathematicalFunction):
    name = "Simionescu Function"
    _name_ = "simionescu_function"
    __name__ = "SimionescuFunction"

    def __init__(
        self,
        A=0.1,
        r_T=1,
        r_S=0.2,
        n=8,
        metric="score",
        input_type="dictionary",
        sleep=0,
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.r_T = r_T
        self.r_S = r_S
        self.n = n

    def objective_function_dict(self, params):
        x = params["x0"].reshape(-1)
        y = params["x1"].reshape(-1)

        condition = (self.r_T + self.r_S * np.cos(self.n * np.arctan(x / y))) ** 2

        mask = x**2 + y**2 <= condition
        mask_int = mask.astype(int)

        loss = self.A * x * y
        loss = mask_int * loss
        loss[~mask] = np.nan

        return self.return_metric(loss)
