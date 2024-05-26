# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class EasomFunction(MathematicalFunction):
    name = "Easom Function"
    _name_ = "easom_function"
    __name__ = "EasomFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x, y) = -\cos (x) \cos (y) \ exp\left[- \left((x-\pi)^2 + (y-\pi)^2 \right) \right]"""
    global_minimum = r"""f(\pi, \pi) = -1"""

    def __init__(self, A=-1, B=1, angle=1, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.angle = angle

    def create_objective_function(self):
        def easom_function(params):
            x = params["x0"]
            y = params["x1"]

            loss1 = self.A * np.cos(x * self.angle) * np.cos(y * self.angle)
            loss2 = np.exp(-((x - np.pi / self.B) ** 2 + (y - np.pi / self.B) ** 2))

            return loss1 * loss2

        self.pure_objective_function = easom_function

    def search_space(self, min=-10, max=10, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
