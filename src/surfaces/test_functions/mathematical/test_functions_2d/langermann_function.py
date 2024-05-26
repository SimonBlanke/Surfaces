# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class LangermannFunction(MathematicalFunction):
    name = "Langermann Function"
    _name_ = "langermann_function"
    __name__ = "LangermannFunction"

    explanation = """
    
    """

    reference = """
    
    """

    dimensions = "2"
    formula = r"""f(\vec x) = \sum^m_{i=1} c_i \exp \left[-\frac{1}{\pi}\sum_{j=1}^d(x_j - A_{ij})^2 \right] \cos \left[\pi \sum_{j=1}^d (x_j - A_{ij})^2 \right]"""
    global_minimum = r"""TODO"""

    c = np.array([1, 2, 5, 2, 3])
    m = 5
    A = np.array([[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]])

    def __init__(self, metric="score", sleep=0):
        super().__init__()
        self.n_dim = 2

    def create_objective_function(self):
        def langermann_function(params):
            loss_sum1 = 0

            for m in range(self.m):
                loss_sum1 += self.c[m]

                loss_sum2 = 0
                loss_sum3 = 0
                for dim in range(self.n_dim):
                    dim_str = "x" + str(dim)
                    x = params[dim_str]

                    loss_sum2 += x - self.A[dim, m]
                    loss_sum3 += x - self.A[dim, m]

                loss_sum2 *= -1 / np.pi
                loss_sum3 *= np.pi

            return loss_sum1 * np.exp(loss_sum2) * np.cos(loss_sum3)

        self.pure_objective_function = langermann_function

    def search_space(self, min=-15, max=15, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
