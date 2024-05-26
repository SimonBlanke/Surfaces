# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class LeviFunctionN13(MathematicalFunction):
    name = "Levi Function N13"
    _name_ = "levi_function_n13"
    __name__ = "LeviFunctionN13"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x,y) = \sin^{2} 3\pi x + \left(x-1\right)^{2}\left(1+\sin^{2} 3\pi y\right)+\left(y-1\right)^{2}\left(1+\sin^{2} 2\pi y\right)"""
    global_minimum = r"""f(1,1)=0"""

    def __init__(self, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

    def create_objective_function(self):
        def levi_function_n13(params):
            x = params["x0"]
            y = params["x1"]

            return (
                np.sin(3 * np.pi * x) ** 2
                + (x + 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
                + (y - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
            )

        self.pure_objective_function = levi_function_n13

    def search_space(self, min=-10, max=10, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
