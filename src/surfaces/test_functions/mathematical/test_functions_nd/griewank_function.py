# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class GriewankFunction(MathematicalFunction):
    name = "Griewank Function"
    _name_ = "griewank_function"
    __name__ = "GriewankFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "n"
    formula = r"""f(\vec x) = \sum^d_{i=1} \frac{x_i^2}{4000} - \prod_{i=1}^d\cos (\frac{x_i}{\sqrt i}) + 1"""
    global_minimum = r"""f(\vec x = 0) = 0"""

    def __init__(self, n_dim, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = n_dim

    def create_objective_function(self):
        def griewank_function(params):
            loss_sum = 0
            loss_product = 1
            for dim in range(self.n_dim):
                dim_str = "x" + str(dim)
                x = params[dim_str]

                loss_sum += x**2 / 4000
                loss_product *= np.cos(x / np.sqrt(dim + 1))

            return loss_sum - loss_product + 1

        self.pure_objective_function = griewank_function

    def search_space(self, min=-100, max=100, size=10000, value_types="array"):
        return super().create_n_dim_search_space(
            min, max, size=size, value_types=value_types
        )
