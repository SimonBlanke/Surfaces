# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_objective_function import MathematicalFunction


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

    def __init__(self, n_dim, metric="score", input_type="dictionary", sleep=0):
        super().__init__()
        self.n_dim = n_dim

    def objective_function_dict(self, params):
        loss_sum = 0
        loss_product = 1
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss_sum += x**2 / 4000
            loss_product *= np.cos(x / np.sqrt(dim + 1))

        loss = loss_sum - loss_product + 1
        return self.return_metric(loss)

    def search_space(self, value_types="array", steps=100):
        min_x0 = -100
        min_x1 = -100

        max_x0 = 100
        max_x1 = 100

        step_size_x0 = (max_x0 - min_x0) / steps
        step_size_x1 = (max_x1 - min_x1) / steps

        return super().search_space(
            search_space_blank={
                "x0": (min_x0, max_x0, step_size_x0),
                "x1": (min_x1, max_x1, step_size_x1),
            },
            value_types=value_types,
        )
