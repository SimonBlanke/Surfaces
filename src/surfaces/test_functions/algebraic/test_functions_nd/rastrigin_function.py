# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .._base_algebraic_function import AlgebraicFunction


class RastriginFunction(AlgebraicFunction):
    """Rastrigin N-dimensional test function.

    A highly multimodal function with many local minima arranged in a
    regular lattice pattern. It is commonly used to test the ability
    of optimization algorithms to escape local optima.

    The function is defined as:

    .. math::

        f(\\vec{x}) = An + \\sum_{i=1}^{n} [x_i^2 - A\\cos(\\omega x_i)]

    where :math:`A = 10` and :math:`\\omega = 2\\pi` by default.

    The global minimum is :math:`f(\\vec{0}) = 0`.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    A : float, default=10
        Amplitude of the cosine modulation.
    angle : float, default=2*pi
        Angular frequency parameter.
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
    >>> from surfaces.test_functions import RastriginFunction
    >>> func = RastriginFunction(n_dim=2)
    >>> result = func({"x0": 0.0, "x1": 0.0})
    >>> abs(result) < 1e-10
    True
    """

    name = "Rastrigin Function"
    _name_ = "rastrigin_function"
    __name__ = "RastriginFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": True,
        "scalable": True,
    }

    f_global = 0.0

    default_bounds = (-5.0, 5.0)

    latex_formula = r"f(\vec{x}) = 10n + \sum_{i=1}^{n} \left[x_i^2 - 10\cos(2\pi x_i)\right]"
    pgfmath_formula = "20 + #1^2 - 10*cos(deg(2*pi*#1)) + #2^2 - 10*cos(deg(2*pi*#2))"  # 2D specialization

    def __init__(
        self,
        n_dim,
        A=10,
        angle=2 * np.pi,
        objective="minimize",
        sleep=0,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
        noise=None,
    ):
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)

        self.n_dim = n_dim
        self.A = A
        self.angle = angle
        self.x_global = np.zeros(n_dim)

    def _create_objective_function(self):
        def rastrigin_function(params):
            loss = 0
            for dim in range(self.n_dim):
                dim_str = "x" + str(dim)
                x = params[dim_str]

                loss += x * x - self.A * np.cos(self.angle * x)

            return self.A * self.n_dim + loss

        self.pure_objective_function = rastrigin_function

    def _search_space(self, min=-5, max=5, size=10000, value_types="array"):
        return super()._create_n_dim_search_space(min, max, size=size, value_types=value_types)
