# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .._base_algebraic_function import AlgebraicFunction


class AckleyFunction(AlgebraicFunction):
    """Ackley two-dimensional test function.

    A non-convex function used as a performance test problem for optimization
    algorithms. It has a nearly flat outer region with a large hole at the
    center, making it challenging for optimization methods.

    The function is defined as:

    .. math::

        f(x, y) = -A \\exp\\left[-0.2\\sqrt{0.5(x^2+y^2)}\\right]
        - \\exp\\left[0.5(\\cos \\omega x + \\cos \\omega y)\\right] + e + A

    where :math:`A = 20` and :math:`\\omega = 2\\pi` by default.

    The global minimum is :math:`f(0, 0) = 0`.

    Parameters
    ----------
    A : float, default=20
        Amplitude parameter.
    angle : float, default=2*pi
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
        Default parameter bounds (-5.0, 5.0).

    References
    ----------
    .. [1] Ackley, D. H. (1987). "A connectionist machine for genetic
       hillclimbing". Kluwer Academic Publishers, Boston MA.

    Examples
    --------
    >>> from surfaces.test_functions import AckleyFunction
    >>> func = AckleyFunction()
    >>> result = func({"x0": 0.0, "x1": 0.0})
    >>> abs(result) < 1e-10
    True
    """

    name = "Ackley Function"
    _name_ = "ackley_function"
    __name__ = "AckleyFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = 0.0
    x_global = np.array([0.0, 0.0])

    default_bounds = (-5.0, 5.0)
    n_dim = 2

    def __init__(
        self,
        A=20,
        angle=2 * np.pi,
        objective="minimize",
        sleep=0,
        memory=False,
        collect_data=True,
        callbacks=None,
    ):
        super().__init__(objective, sleep, memory, collect_data, callbacks)

        self.n_dim = 2

        self.A = A
        self.angle = angle

    def _create_objective_function(self):
        def ackley_function(params):
            x = params["x0"]
            y = params["x1"]

            loss1 = -self.A * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
            loss2 = -np.exp(0.5 * (np.cos(self.angle * x) + np.cos(self.angle * y)))
            loss3 = np.exp(1)
            loss4 = self.A

            return loss1 + loss2 + loss3 + loss4

        self.pure_objective_function = ackley_function

    def _search_space(self, min=-5, max=5, value_types="array", size=10000):
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
