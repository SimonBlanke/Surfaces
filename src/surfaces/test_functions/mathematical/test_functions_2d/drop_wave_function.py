# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class DropWaveFunction(MathematicalFunction):
    name = "Drop Wave Function"
    _name_ = "drop_wave_function"
    __name__ = "DropWaveFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x, y) = -\frac{1+\cos (12\sqrt{x^2+y^2})}{0.5 (x^2 + y^2) + 2}"""
    global_minimum = r"""f(0, 0)= -1"""

    def __init__(self, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

    def create_objective_function(self):
        def drop_wave_function(params):
            x = params["x0"]
            y = params["x1"]

            return -(1 + np.cos(12 * np.sqrt(x**2 + y**2))) / (0.5 * (x**2 + y**2) + 2)

        self.pure_objective_function = drop_wave_function

    def search_space(self, min=-5, max=5, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
