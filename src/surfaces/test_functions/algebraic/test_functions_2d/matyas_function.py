# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .._base_algebraic_function import AlgebraicFunction


class MatyasFunction(AlgebraicFunction):
    """Matyas two-dimensional test function.

    A bowl-shaped, unimodal function.

    The function is defined as:

    .. math::

        f(x, y) = 0.26(x^2 + y^2) - 0.48xy

    The global minimum is :math:`f(0, 0) = 0`.

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
    >>> from surfaces.test_functions import MatyasFunction
    >>> func = MatyasFunction()
    >>> result = func({"x0": 0.0, "x1": 0.0})
    >>> abs(result) < 1e-10
    True
    """

    name = "Matyas Function"
    _name_ = "matyas_function"
    __name__ = "MatyasFunction"

    _spec = {
        "convex": True,
        "unimodal": True,
        "separable": False,
        "scalable": False,
    }

    f_global = 0.0
    x_global = np.array([0.0, 0.0])

    default_bounds = (-10.0, 10.0)
    n_dim = 2

    def __init__(self, objective="minimize", sleep=0, memory=False, collect_data=True, callbacks=None, catch_errors=None):
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors)
        self.n_dim = 2

    def _create_objective_function(self):
        def matyas_function(params):
            x = params["x0"]
            y = params["x1"]

            return 0.26 * (x**2 + y**2) - 0.48 * x * y

        self.pure_objective_function = matyas_function

    def _search_space(self, min=-10, max=10, value_types="array", size=10000):
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
