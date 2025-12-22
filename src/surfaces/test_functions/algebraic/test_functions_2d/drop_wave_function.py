# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .._base_algebraic_function import AlgebraicFunction


class DropWaveFunction(AlgebraicFunction):
    """Drop-Wave two-dimensional test function.

    A highly multimodal function with many local minima arranged in a
    concentric wave pattern.

    The function is defined as:

    .. math::

        f(x, y) = -\\frac{1 + \\cos(12\\sqrt{x^2 + y^2})}{0.5(x^2 + y^2) + 2}

    The global minimum is :math:`f(0, 0) = -1`.

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
    >>> from surfaces.test_functions import DropWaveFunction
    >>> func = DropWaveFunction()
    >>> result = func({"x0": 0.0, "x1": 0.0})
    >>> abs(result + 1.0) < 1e-10
    True
    """

    name = "Drop Wave Function"
    _name_ = "drop_wave_function"
    __name__ = "DropWaveFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = -1.0
    x_global = np.array([0.0, 0.0])

    default_bounds = (-5.0, 5.0)
    n_dim = 2

    def __init__(self, objective="minimize", sleep=0, memory=False, collect_data=True):
        super().__init__(objective, sleep, memory, collect_data)
        self.n_dim = 2

    def _create_objective_function(self):
        def drop_wave_function(params):
            x = params["x0"]
            y = params["x1"]

            return -(1 + np.cos(12 * np.sqrt(x**2 + y**2))) / (0.5 * (x**2 + y**2) + 2)

        self.pure_objective_function = drop_wave_function

    def _search_space(self, min=-5, max=5, value_types="array", size=10000):
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
