# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import MathematicalFunction


class GriewankFunction(MathematicalFunction):
    name = "Griewank Function"
    _name_ = "griewank_function"
    __name__ = "GriewankFunction"

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
