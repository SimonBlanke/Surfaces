# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_mathematical_function import MathematicalFunction


class SchafferFunctionN2(MathematicalFunction):
    """Schaffer N.2 two-dimensional test function.

    A multimodal function with many local minima.

    The function is defined as:

    .. math::

        f(x, y) = 0.5 + \\frac{\\sin^2(x^2 - y^2) - 0.5}
        {[1 + 0.001(x^2 + y^2)]^2}

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
        Default parameter bounds (-50.0, 50.0).

    Examples
    --------
    >>> from surfaces.test_functions import SchafferFunctionN2
    >>> func = SchafferFunctionN2()
    >>> result = func({"x0": 0.0, "x1": 0.0})
    """

    name = "Schaffer Function N2"
    _name_ = "schaffer_function_n2"
    __name__ = "SchafferFunctionN2"

    default_bounds = (-50.0, 50.0)

    def __init__(self, metric="score", sleep=0):
        super().__init__(metric, sleep)
        self.n_dim = 2

    def _create_objective_function(self):
        def schaffer_function_n2(params):
            x = params["x0"]
            y = params["x1"]

            return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

        self.pure_objective_function = schaffer_function_n2

    def _search_space(self, min=-50, max=50, value_types="array", size=10000):
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
