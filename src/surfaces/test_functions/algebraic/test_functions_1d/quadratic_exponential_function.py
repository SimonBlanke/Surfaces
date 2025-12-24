# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_algebraic_function import AlgebraicFunction


class QuadraticExponentialFunction(AlgebraicFunction):
    """Quadratic-Exponential one-dimensional test function.

    A one-dimensional test function combining a quadratic polynomial
    with an exponential decay term. The function has a smooth,
    single-valley landscape.

    The function is defined as:

    .. math::

        f(x) = -(16x^2 - 24x + 5) e^{-x}

    Parameters
    ----------
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    n_dim : int
        Number of dimensions (always 1).
    default_bounds : tuple
        Default parameter bounds (1.9, 3.9).

    References
    ----------
    .. [1] AMPGO (Adaptive Memory Programming for Global Optimization)
       benchmark suite, Problem04.
       http://infinity77.net/global_optimization/test_functions_1d.html

    .. [2] Gavana, A. (2013). "Global Optimization Benchmarks and AMPGO".

    Examples
    --------
    >>> from surfaces.test_functions import QuadraticExponentialFunction
    >>> func = QuadraticExponentialFunction()
    >>> func({"x0": 2.868})  # Near global minimum
    -3.850...
    >>> search_space = func.search_space
    >>> len(search_space)
    1
    """

    name = "Quadratic Exponential Function"
    _name_ = "quadratic_exponential_function"
    __name__ = "QuadraticExponentialFunction"

    _spec = {
        "convex": False,
        "unimodal": True,
        "separable": True,
        "scalable": False,
    }

    f_global = -3.8504536747755516
    x_global = np.array([2.8680336666498003])

    default_bounds = (1.9, 3.9)
    n_dim = 1

    latex_formula = r"f(x) = -(16x^2 - 24x + 5) e^{-x}"
    pgfmath_formula = "-(16*#1^2 - 24*#1 + 5) * exp(-#1)"

    def __init__(
        self,
        objective="minimize",
        sleep=0,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
        noise=None,
    ):
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)
        self.n_dim = 1

    def _create_objective_function(self):
        def quadratic_exponential_function(params):
            x = params["x0"]

            return -(16 * x**2 - 24 * x + 5) * np.exp(-x)

        self.pure_objective_function = quadratic_exponential_function

    def _search_space(self, min=1.9, max=3.9, value_types="array", size=10000):
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
