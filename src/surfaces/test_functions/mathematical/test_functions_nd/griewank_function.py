# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class GriewankFunction(MathematicalFunction):
    """Griewank N-dimensional test function.

    A multimodal function with many regularly distributed local minima.
    The number of local minima increases with dimensionality.

    The function is defined as:

    .. math::

        f(\\vec{x}) = \\sum_{i=1}^{n} \\frac{x_i^2}{4000}
        - \\prod_{i=1}^{n} \\cos\\left(\\frac{x_i}{\\sqrt{i}}\\right) + 1

    The global minimum is :math:`f(\\vec{0}) = 0`.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    default_bounds : tuple
        Default parameter bounds (-100.0, 100.0).

    Examples
    --------
    >>> from surfaces.test_functions import GriewankFunction
    >>> func = GriewankFunction(n_dim=2)
    >>> result = func({"x0": 0.0, "x1": 0.0})
    >>> abs(result) < 1e-10
    True
    """

    name = "Griewank Function"
    _name_ = "griewank_function"
    __name__ = "GriewankFunction"

    default_bounds = (-100.0, 100.0)

    def __init__(self, n_dim, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = n_dim

    def _create_objective_function(self):
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

    def _search_space(self, min=-100, max=100, size=10000, value_types="array"):
        return super()._create_n_dim_search_space(
            min, max, size=size, value_types=value_types
        )
