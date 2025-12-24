# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .._base_algebraic_function import AlgebraicFunction


class ForresterFunction(AlgebraicFunction):
    """Forrester one-dimensional test function.

    A one-dimensional test function commonly used for testing surrogate
    modeling and multi-fidelity optimization methods. The function is
    multimodal with one global minimum, one local minimum, and a
    zero-gradient inflection point.

    The function is defined as:

    .. math::

        f(x) = (6x - 2)^2 \\sin(12x - 4)

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
        Default parameter bounds (0, 1).

    References
    ----------
    .. [1] Forrester, A., Sobester, A., & Keane, A. (2008). "Engineering
       design via surrogate modelling: a practical guide". Wiley.

    .. [2] Virtual Library of Simulation Experiments (VLSE):
       https://www.sfu.ca/~ssurjano/forretal08.html

    Examples
    --------
    >>> from surfaces.test_functions import ForresterFunction
    >>> func = ForresterFunction()
    >>> func({"x0": 0.757249})  # Near global minimum
    -6.020740...
    >>> search_space = func.search_space
    >>> len(search_space)
    1
    """

    name = "Forrester Function"
    _name_ = "forrester_function"
    __name__ = "ForresterFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": True,
        "scalable": False,
    }

    f_global = -6.020740095378691
    x_global = np.array([0.7572489953693498])

    default_bounds = (0.0, 1.0)
    n_dim = 1

    latex_formula = r"f(x) = (6x - 2)^2 \sin(12x - 4)"
    pgfmath_formula = "(6*#1 - 2)^2 * sin(deg(12*#1 - 4))"

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
        def forrester_function(params):
            x = params["x0"]

            return ((6 * x - 2) ** 2) * np.sin(12 * x - 4)

        self.pure_objective_function = forrester_function

    def _search_space(self, min=0.0, max=1.0, value_types="array", size=10000):
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
