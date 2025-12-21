# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .._base_algebraic_function import AlgebraicFunction


class StyblinskiTangFunction(AlgebraicFunction):
    """Styblinski-Tang N-dimensional test function.

    A polynomial function with multiple local minima.

    The function is defined as:

    .. math::

        f(\\vec{x}) = \\frac{1}{2} \\sum_{i=1}^{n} (x_i^4 - 16x_i^2 + 5x_i)

    The global minimum is approximately
    :math:`f(-2.903534, ..., -2.903534) \\approx -39.16617n`.

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
        Default parameter bounds (-5.0, 5.0).

    Examples
    --------
    >>> from surfaces.test_functions import StyblinskiTangFunction
    >>> func = StyblinskiTangFunction(n_dim=2)
    >>> result = func({"x0": -2.903534, "x1": -2.903534})
    """

    name = "Styblinski Tang Function"
    _name_ = "styblinski_tang_function"
    __name__ = "StyblinskiTangFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": True,
        "scalable": True,
    }

    default_bounds = (-5.0, 5.0)

    def __init__(self, n_dim, objective="minimize", sleep=0):
        super().__init__(objective, sleep)
        self.n_dim = n_dim
        self.x_global = np.full(n_dim, -2.903534)
        self.f_global = -39.16617 * n_dim

    def _create_objective_function(self):
        def styblinski_tang_function(params):
            loss = 0
            for dim in range(self.n_dim):
                dim_str = "x" + str(dim)
                x = params[dim_str]

                loss += x**4 - 16 * x**2 + 5 * x

            return loss / 2

        self.pure_objective_function = styblinski_tang_function

    def _search_space(self, min=-5, max=5, size=10000, value_types="array"):
        return super()._create_n_dim_search_space(min, max, size=size, value_types=value_types)
