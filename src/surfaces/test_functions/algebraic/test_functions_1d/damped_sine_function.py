# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_algebraic_function import AlgebraicFunction


class DampedSineFunction(AlgebraicFunction):
    """Damped Sine one-dimensional test function.

    A one-dimensional test function combining sine and linear terms
    with Gaussian damping. The exponential decay creates a localized
    oscillatory region near the origin.

    The function is defined as:

    .. math::

        f(x) = -(x + \\sin(x)) e^{-x^2}

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
        Default parameter bounds (-10, 10).

    References
    ----------
    .. [1] AMPGO (Adaptive Memory Programming for Global Optimization)
       benchmark suite, Problem06.
       http://infinity77.net/global_optimization/test_functions_1d.html

    .. [2] Gavana, A. (2013). "Global Optimization Benchmarks and AMPGO".

    Examples
    --------
    >>> from surfaces.test_functions import DampedSineFunction
    >>> func = DampedSineFunction()
    >>> func({"x0": 0.6796})  # Near global minimum
    -0.824...
    >>> search_space = func.search_space
    >>> len(search_space)
    1
    """

    name = "Damped Sine Function"
    _name_ = "damped_sine_function"
    __name__ = "DampedSineFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": True,
        "scalable": False,
    }

    f_global = -0.8242393984760573
    x_global = np.array([0.6795787635255166])

    default_bounds = (-10.0, 10.0)
    n_dim = 1

    latex_formula = r"f(x) = -(x + \sin(x)) e^{-x^2}"
    pgfmath_formula = "-(#1 + sin(deg(#1))) * exp(-#1^2)"

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
        def damped_sine_function(params):
            x = params["x0"]

            return -(x + np.sin(x)) * np.exp(-(x**2))

        self.pure_objective_function = damped_sine_function

    def _search_space(self, min=-10.0, max=10.0, value_types="array", size=10000):
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
