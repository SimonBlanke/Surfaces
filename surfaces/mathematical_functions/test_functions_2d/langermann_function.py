# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_objective_function import MathematicalFunction


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

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__()
        self.n_dim = 2

    def objective_function_dict(self, params):
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

        loss = loss_sum1 * np.exp(loss_sum2) * np.cos(loss_sum3)
        return self.return_metric(loss)

    def search_space(self, value_types="array", steps=100):
        min_x0 = -15
        min_x1 = -15

        max_x0 = 15
        max_x1 = 15

        step_size_x0 = (max_x0 - min_x0) / steps
        step_size_x1 = (max_x1 - min_x1) / steps

        return super().search_space(
            search_space_blank={
                "x0": (min_x0, max_x0, step_size_x0),
                "x1": (min_x1, max_x1, step_size_x1),
            },
            value_types=value_types,
        )
