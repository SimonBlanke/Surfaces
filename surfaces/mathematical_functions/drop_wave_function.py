# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import MathematicalFunction


class DropWaveFunction(MathematicalFunction):
    name = "Drop Wave Function"
    _name_ = "drop_wave_function"
    __name__ = "DropWaveFunction"

    dimensions = "2"
    formula = r"""f(x, y) = -\frac{1+\cos (12\sqrt{x^2+y^2})}{0.5 (x^2 + y^2) + 2}"""
    global_minimum = r"""f(0, 0)= -1"""

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = -(1 + np.cos(12 * np.sqrt(x**2 + y**2))) / (
            0.5 * (x**2 + y**2) + 2
        )
        return self.return_metric(loss)
