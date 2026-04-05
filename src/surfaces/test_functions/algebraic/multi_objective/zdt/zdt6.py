# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""ZDT6 multi-objective test function."""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from .._base_multi_objective import BaseMultiObjectiveTestFunction


class ZDT6(BaseMultiObjectiveTestFunction):
    """ZDT6 multi-objective test function.

    The ZDT6 function features a non-uniform density of solutions across
    the Pareto front, with solutions being more sparse near the front
    and denser away from it. This makes it particularly challenging for
    algorithms that maintain diversity.

    The function is defined as:

    .. math::

        f_1(x) = 1 - \\exp(-4 x_1) \\sin^6(6 \\pi x_1)

        f_2(x) = g(x) \\cdot \\left[1 - \\left(\\frac{f_1(x)}{g(x)}\\right)^2\\right]

        g(x) = 1 + 9 \\left[\\frac{\\sum_{i=2}^{n} x_i}{n-1}\\right]^{0.25}

    Parameters
    ----------
    n_dim : int, default=10
        Number of input dimensions. Must be >= 2.
    **kwargs
        Additional keyword arguments passed to
        :class:`BaseMultiObjectiveTestFunction` (modifiers, memory, etc.).

    Attributes
    ----------
    n_objectives : int
        Number of objectives (always 2).

    References
    ----------
    .. [1] Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of
       multiobjective evolutionary algorithms: Empirical results.
       Evolutionary computation, 8(2), 173-195.

    Examples
    --------
    >>> from surfaces.test_functions.algebraic.multi_objective.zdt import ZDT6
    >>> func = ZDT6(n_dim=10)
    >>> result = func(np.zeros(10))
    >>> result.shape
    (2,)
    """

    name = "ZDT6"
    n_objectives = 2
    _spec = {
        "eval_cost": 1.3,
        "continuous": True,
        "differentiable": True,
        "concave_front": True,
        "scalable": True,
        "default_bounds": (0.0, 1.0),
    }

    def __init__(self, n_dim: int = 10, **kwargs):
        if n_dim < 2:
            raise ValueError(f"n_dim must be >= 2, got {n_dim}")
        super().__init__(n_dim, **kwargs)

    @staticmethod
    def _f1_from_x0(x0):
        """Compute f1 from x0, used by both _objective and _pareto_front."""
        return 1 - np.exp(-4 * x0) * np.sin(6 * np.pi * x0) ** 6

    def _objective(self, params: Dict[str, Any]) -> np.ndarray:
        x = self._params_to_array(params)

        f1 = self._f1_from_x0(x[0])

        g = 1 + 9 * (np.sum(x[1:]) / (self.n_dim - 1)) ** 0.25
        f2 = g * (1 - (f1 / g) ** 2)

        return np.array([f1, f2])

    def _pareto_front(self, n_points: int) -> np.ndarray:
        """Pareto front: f2 = 1 - f1^2 with f1 starting at its minimum."""
        # f1_min is the minimum of 1 - exp(-4*x0)*sin(6*pi*x0)^6 over [0,1]
        x0_dense = np.linspace(0, 1, 10000)
        f1_dense = self._f1_from_x0(x0_dense)
        f1_min = f1_dense.min()

        f1 = np.linspace(f1_min, 1, n_points)
        f2 = 1 - f1**2
        return np.column_stack([f1, f2])

    def _pareto_set(self, n_points: int) -> np.ndarray:
        """Pareto set: x1 in [0, 1] with x2 = ... = xn = 0."""
        x = np.zeros((n_points, self.n_dim))
        x[:, 0] = np.linspace(0, 1, n_points)
        return x

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized ZDT6 evaluation."""
        xp = get_array_namespace(X)

        f1 = 1 - xp.exp(-4 * X[:, 0]) * xp.sin(6 * np.pi * X[:, 0]) ** 6
        g = 1 + 9 * (xp.sum(X[:, 1:], axis=1) / (self.n_dim - 1)) ** 0.25
        f2 = g * (1 - (f1 / g) ** 2)

        return xp.stack([f1, f2], axis=1)
