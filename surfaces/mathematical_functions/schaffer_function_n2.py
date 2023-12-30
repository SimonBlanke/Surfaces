# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import MathematicalFunction


class SchafferFunctionN2(MathematicalFunction):
    name = "Schaffer Function N2"
    _name_ = "schaffer_function_n2"
    __name__ = "SchafferFunctionN2"

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

        return self.return_metric(loss)
