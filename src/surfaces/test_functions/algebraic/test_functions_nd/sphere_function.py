# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .._base_algebraic_function import AlgebraicFunction


class SphereFunction(AlgebraicFunction):
    """Sphere N-dimensional test function.

    A continuous, convex, and unimodal function. It is the simplest
    N-dimensional optimization test function.

    The function is defined as:

    .. math::

        f(\\vec{x}) = A \\sum_{i=1}^{n} x_i^2

    where :math:`A = 1` by default.

    The global minimum is :math:`f(\\vec{0}) = 0`.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    A : float, default=1
        Scaling parameter.
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.
    validate : bool, default=True
        Whether to validate parameters against the search space.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    default_bounds : tuple
        Default parameter bounds (-5.0, 5.0).

    Examples
    --------
    >>> from surfaces.test_functions import SphereFunction
    >>> func = SphereFunction(n_dim=3)
    >>> result = func({"x0": 0.0, "x1": 0.0, "x2": 0.0})
    >>> abs(result) < 1e-10
    True
    >>> len(func.search_space)
    3
    """

    name = "Sphere Function"
    _name_ = "sphere_function"
    __name__ = "SphereFunction"

    _spec = {
        "convex": True,
        "unimodal": True,
        "separable": True,
        "scalable": True,
    }

    f_global = 0.0

    default_bounds = (-5.0, 5.0)

    def __init__(self, n_dim, A=1, objective="minimize", sleep=0, memory=False, collect_data=True, callbacks=None):
        super().__init__(objective, sleep, memory, collect_data, callbacks)
        self.n_dim = n_dim
        self.A = A
        self.x_global = np.zeros(n_dim)

    def _create_objective_function(self):
        def sphere_function(params):
            loss = 0
            for dim in range(self.n_dim):
                dim_str = "x" + str(dim)
                x = params[dim_str]

                loss += self.A * x * x

            return loss

        self.pure_objective_function = sphere_function

    def _search_space(self, min=-5, max=5, size=10000, value_types="array"):
        return super()._create_n_dim_search_space(min, max, size=size, value_types=value_types)
