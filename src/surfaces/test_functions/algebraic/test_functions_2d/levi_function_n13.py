# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .._base_algebraic_function import AlgebraicFunction


class LeviFunctionN13(AlgebraicFunction):
    """Levi N.13 two-dimensional test function.

    A multimodal function with a single global minimum.

    The function is defined as:

    .. math::

        f(x, y) = \\sin^2(3\\pi x) + (x-1)^2(1 + \\sin^2(3\\pi y))
        + (y-1)^2(1 + \\sin^2(2\\pi y))

    The global minimum is :math:`f(1, 1) = 0`.

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
    default_bounds : tuple
        Default parameter bounds (-10.0, 10.0).

    Examples
    --------
    >>> from surfaces.test_functions import LeviFunctionN13
    >>> func = LeviFunctionN13()
    >>> result = func({"x0": 1.0, "x1": 1.0})
    """

    name = "Levi Function N13"
    _name_ = "levi_function_n13"
    __name__ = "LeviFunctionN13"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = 0.0
    x_global = np.array([1.0, 1.0])

    default_bounds = (-10.0, 10.0)
    n_dim = 2

    def __init__(self, objective="minimize", sleep=0):
        super().__init__(objective, sleep)
        self.n_dim = 2

    def _create_objective_function(self):
        def levi_function_n13(params):
            x = params["x0"]
            y = params["x1"]

            return (
                np.sin(3 * np.pi * x) ** 2
                + (x + 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
                + (y - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
            )

        self.pure_objective_function = levi_function_n13

    def _search_space(self, min=-10, max=10, value_types="array", size=10000):
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
