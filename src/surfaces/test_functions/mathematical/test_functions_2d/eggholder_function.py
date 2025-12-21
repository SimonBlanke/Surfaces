# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .._base_mathematical_function import MathematicalFunction


class EggholderFunction(MathematicalFunction):
    """Eggholder two-dimensional test function.

    A difficult optimization test problem due to its many local minima.

    The function is defined as:

    .. math::

        f(x, y) = -(y + 47) \\sin\\sqrt{|\\frac{x}{2} + (y + 47)|}
        - x \\sin\\sqrt{|x - (y + 47)|}

    The global minimum is :math:`f(512, 404.2319) = -959.6407`.

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
        Default parameter bounds (-1000.0, 1000.0).

    Examples
    --------
    >>> from surfaces.test_functions import EggholderFunction
    >>> func = EggholderFunction()
    >>> result = func({"x0": 512.0, "x1": 404.2319})
    """

    name = "Eggholder Function"
    _name_ = "eggholder_function"
    __name__ = "EggholderFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = -959.6407
    x_global = np.array([512.0, 404.2319])

    default_bounds = (-1000.0, 1000.0)
    n_dim = 2

    def __init__(self, objective="minimize", sleep=0):
        super().__init__(objective, sleep)
        self.n_dim = 2

    def _create_objective_function(self):
        def eggholder_function(params):
            x = params["x0"]
            y = params["x1"]

            return (y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(
                np.sqrt(np.abs(x - (y + 47)))
            )

        self.pure_objective_function = eggholder_function

    def _search_space(self, min=-1000, max=1000, value_types="array", size=10000):
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
