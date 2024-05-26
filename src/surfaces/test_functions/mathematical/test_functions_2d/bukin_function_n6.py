# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class BukinFunctionN6(MathematicalFunction):
    name = "Bukin Function N6"
    _name_ = "bukin_function_n6"
    __name__ = "BukinFunctionN6"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = (
        r"""f(x,y) = 100 \sqrt{\left| y-0.01x^2 \right|} + 0.01 \left|x+10\right|"""
    )
    global_minimum = r"""f(-10,1)=0"""

    def __init__(self, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

    def create_objective_function(self):
        def bukin_function_n6(params):
            x = params["x0"]
            y = params["x1"]

            return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

        self.pure_objective_function = bukin_function_n6

    def search_space(self, min=-8, max=8, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
