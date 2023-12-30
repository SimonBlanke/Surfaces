# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import ObjectiveFunction


class McCormickFunction(ObjectiveFunction):
    name = "Mc Cormick Function"
    _name_ = "mccormick_function"
    __name__ = "McCormickFunction"

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

        return self.return_metric(loss)
