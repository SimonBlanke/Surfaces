# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class SchafferFunctionN2(MathematicalFunction):
    name = "Schaffer Function N2"
    _name_ = "schaffer_function_n2"
    __name__ = "SchafferFunctionN2"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x,y) = 0.5 + \frac{\sin^{2}\left(x^{2} - y^{2}\right) - 0.5}{\left[1 + 0.001\left(x^{2} + y^{2}\right) \right]^{2}}"""
    global_minimum = r"""f(0,0) = 0"""

    def __init__(self, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

    def create_objective_function(self):
        def schaffer_function_n2(params):
            x = params["x0"]
            y = params["x1"]

            return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

        self.pure_objective_function = schaffer_function_n2

    def search_space(self, min=-50, max=50, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
