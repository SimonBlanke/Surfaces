# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class LangermannFunction(MathematicalFunction):
    """Langermann two-dimensional test function.

    A multimodal function with many unevenly distributed local minima.

    Parameters
    ----------
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    n_dim : int
        Number of dimensions (always 2).
    c : ndarray
        Coefficient vector.
    m : int
        Number of terms in the summation.
    A : ndarray
        Matrix of center coordinates.

    Examples
    --------
    >>> from surfaces.test_functions import LangermannFunction
    >>> func = LangermannFunction()
    >>> result = func({"x0": 0.0, "x1": 0.0})
    """

    name = "Langermann Function"
    _name_ = "langermann_function"
    __name__ = "LangermannFunction"

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

    def _search_space(self, min=-15, max=15, value_types="array", size=10000):
        return super().create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
