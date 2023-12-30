# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import MathematicalFunction


class BukinFunctionN6(MathematicalFunction):
    name = "Bukin Function N6"
    _name_ = "bukin_function_n6"
    __name__ = "BukinFunctionN6"

    dimensions = "2"
    formula = (
        r"""f(x,y) = 100 \sqrt{\left| y-0.01x^2 \right|} + 0.01 \left|x+10\right|"""
    )
    global_minimum = r"""f(-10,1)=0"""

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

        return self.return_metric(loss)
