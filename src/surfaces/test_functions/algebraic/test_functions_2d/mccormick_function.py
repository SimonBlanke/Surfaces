# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .._base_algebraic_function import AlgebraicFunction


class McCormickFunction(AlgebraicFunction):
    """McCormick two-dimensional test function.

    A function with a single global minimum, commonly used for testing
    optimization algorithms.

    The function is defined as:

    .. math::

        f(x, y) = \\sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1

    The global minimum is :math:`f(-0.54719, -1.54719) = -1.9133`.

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
        Default parameter bounds (-5.0, 5.0).

    Examples
    --------
    >>> from surfaces.test_functions import McCormickFunction
    >>> func = McCormickFunction()
    >>> result = func({"x0": -0.54719, "x1": -1.54719})
    """

    name = "Mc Cormick Function"
    _name_ = "mccormick_function"
    __name__ = "McCormickFunction"

    _spec = {
        "convex": False,
        "unimodal": True,
        "separable": False,
        "scalable": False,
    }

    f_global = -1.9133
    x_global = np.array([-0.54719, -1.54719])

    default_bounds = (-5.0, 5.0)
    n_dim = 2

    latex_formula = r"f(x, y) = \sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1"
    pgfmath_formula = "sin(deg(#1 + #2)) + (#1 - #2)^2 - 1.5*#1 + 2.5*#2 + 1"

    def __init__(self, objective="minimize", sleep=0, memory=False, collect_data=True, callbacks=None, catch_errors=None):
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors)
        self.n_dim = 2

    def _create_objective_function(self):
        def mccormick_function(params):
            x = params["x0"]
            y = params["x1"]

            return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

        self.pure_objective_function = mccormick_function

    def _search_space(self, min=-5, max=5, value_types="array", size=10000):
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
