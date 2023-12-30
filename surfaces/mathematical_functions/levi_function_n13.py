# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import MathematicalFunction


class LeviFunctionN13(MathematicalFunction):
    name = "Levi Function N13"
    _name_ = "levi_function_n13"
    __name__ = "LeviFunctionN13"

    dimensions = "2"
    formula = r"""f(x,y) = \sin^{2} 3\pi x + \left(x-1\right)^{2}\left(1+\sin^{2} 3\pi y\right)+\left(y-1\right)^{2}\left(1+\sin^{2} 2\pi y\right)"""
    global_minimum = r"""f(1,1)=0"""

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = (
            np.sin(3 * np.pi * x) ** 2
            + (x + 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
            + (y - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
        )

        return self.return_metric(loss)
