# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class ThreeHumpCamelFunction(MathematicalFunction):
    name = "Three Hump Camel Function"
    _name_ = "three_hump_camel_function"
    __name__ = "ThreeHumpCamelFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(x,y) = -\cos \left(x\right)\cos \left(y\right) \exp\left[-\left(\left(x-\pi\right)^{2} + \left(y-\pi\right)^{2}\right)\right]"""
    global_minimum = r"""f(0,0)= 0"""

    def __init__(self, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

    def create_objective_function(self):
        def three_hump_camel_function(params):
            x = params["x0"]
            y = params["x1"]

            return 2 * x**2 - 1.05 * x**4 + x**6 / 6 + x * y + y**2

        self.pure_objective_function = three_hump_camel_function

    def search_space(self, min=-5, max=5, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
