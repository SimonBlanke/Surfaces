# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ._base_objective_function import ObjectiveFunction


class LangermannFunction(ObjectiveFunction):
    name = "Langermann Function"
    _name_ = "langermann_function"
    __name__ = "LangermannFunction"

    c = np.array([1, 2, 5, 2, 3])
    m = 5
    A = np.array([[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]])

    def __init__(self, *args, **kwargs):
        super().__init__(args, **kwargs)
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