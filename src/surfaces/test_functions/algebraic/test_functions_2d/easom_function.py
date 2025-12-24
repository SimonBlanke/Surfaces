# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .._base_algebraic_function import AlgebraicFunction


class EasomFunction(AlgebraicFunction):
    """Easom two-dimensional test function.

    A unimodal function with a very small area relative to the search space
    where the function has a significant gradient. The global minimum has
    a small basin of attraction.

    The function is defined as:

    .. math::

        f(x, y) = A \\cos(\\omega x) \\cos(\\omega y)
        \\exp\\left[-(x - \\pi/B)^2 - (y - \\pi/B)^2\\right]

    where :math:`A = -1`, :math:`B = 1`, and :math:`\\omega = 1` by default.

    The global minimum is :math:`f(\\pi, \\pi) = -1`.

    Parameters
    ----------
    A : float, default=-1
        Amplitude parameter.
    B : float, default=1
        Scaling parameter for the optimum location.
    angle : float, default=1
        Angular frequency parameter.
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
    >>> from surfaces.test_functions import EasomFunction
    >>> import numpy as np
    >>> func = EasomFunction()
    >>> result = func({"x0": np.pi, "x1": np.pi})
    >>> abs(result + 1.0) < 1e-10
    True
    """

    name = "Easom Function"
    _name_ = "easom_function"
    __name__ = "EasomFunction"

    _spec = {
        "convex": False,
        "unimodal": True,
        "separable": True,
        "scalable": False,
    }

    f_global = -1.0
    x_global = np.array([np.pi, np.pi])

    default_bounds = (-10.0, 10.0)
    n_dim = 2

    latex_formula = r"f(x, y) = -\cos(x)\cos(y)\exp\left[-(x - \pi)^2 - (y - \pi)^2\right]"
    pgfmath_formula = "-cos(deg(#1))*cos(deg(#2))*exp(-((#1 - pi)^2 + (#2 - pi)^2))"

    def __init__(
        self,
        A=-1,
        B=1,
        angle=1,
        objective="minimize",
        sleep=0,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
        noise=None,
    ):
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.angle = angle

    def _create_objective_function(self):
        def easom_function(params):
            x = params["x0"]
            y = params["x1"]

            loss1 = self.A * np.cos(x * self.angle) * np.cos(y * self.angle)
            loss2 = np.exp(-((x - np.pi / self.B) ** 2 + (y - np.pi / self.B) ** 2))

            return loss1 * loss2

        self.pure_objective_function = easom_function

    def _search_space(self, min=-10, max=10, value_types="array", size=10000):
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
