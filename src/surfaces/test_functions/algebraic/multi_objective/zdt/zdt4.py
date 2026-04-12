# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""ZDT4 multi-objective test function."""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from .._base_multi_objective import BaseMultiObjectiveTestFunction


class ZDT4(BaseMultiObjectiveTestFunction):
    """ZDT4 multi-objective test function.

    The ZDT4 function tests an optimizer's ability to deal with
    multimodality. The g function introduces many local Pareto-optimal
    fronts, making convergence to the global Pareto front difficult.

    The function is defined as:

    .. math::

        f_1(x) = x_1

        f_2(x) = g(x) \\cdot \\left[1 - \\sqrt{\\frac{x_1}{g(x)}}\\right]

        g(x) = 1 + 10(n-1) + \\sum_{i=2}^{n}
               \\left[x_i^2 - 10 \\cos(4 \\pi x_i)\\right]

    where :math:`x_1 \\in [0, 1]` and :math:`x_i \\in [-5, 5]` for
    :math:`i = 2, \\ldots, n`.

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
    >>> from surfaces.test_functions.algebraic.multi_objective.zdt import ZDT4
    >>> func = ZDT4(n_dim=10)
    >>> result = func(np.zeros(10))
    >>> result.shape
    (2,)
    >>> result[0]  # f1 = x1 = 0
    0.0
    """

    name = "ZDT4"
    n_objectives = 2
    _spec = {
        "eval_cost": 1.5,
        "continuous": True,
        "differentiable": True,
        "convex_front": True,
        "multimodal": True,
        "scalable": True,
        "default_bounds": (0.0, 1.0),
    }

    def __init__(self, n_dim: int = 10, **kwargs):
        if n_dim < 2:
            raise ValueError(f"n_dim must be >= 2, got {n_dim}")
        super().__init__(n_dim, **kwargs)

    def _default_search_space(self) -> Dict[str, Any]:
        """Non-uniform bounds: x0 in [0,1], x1..xn in [-5,5]."""
        dim_size = int(self.default_size ** (1 / self.n_dim))

        search_space = {}

        step_x0 = 1.0 / dim_size
        search_space["x0"] = np.arange(0.0, 1.0, step_x0)

        step_xi = 10.0 / dim_size
        for dim in range(1, self.n_dim):
            search_space[f"x{dim}"] = np.arange(-5.0, 5.0, step_xi)

        return search_space

    def _objective(self, params: Dict[str, Any]) -> np.ndarray:
        x = self._params_to_array(params)

        f1 = x[0]

        g = 1 + 10 * (self.n_dim - 1) + np.sum(x[1:] ** 2 - 10 * np.cos(4 * np.pi * x[1:]))
        f2 = g * (1 - np.sqrt(f1 / g))

        return np.array([f1, f2])

    def _pareto_front(self, n_points: int) -> np.ndarray:
        """Pareto front: f2 = 1 - sqrt(f1), achieved when g=1 (x[1:]=0)."""
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        return np.column_stack([f1, f2])

    def _pareto_set(self, n_points: int) -> np.ndarray:
        """Pareto set: x1 in [0, 1] with x2 = ... = xn = 0."""
        x = np.zeros((n_points, self.n_dim))
        x[:, 0] = np.linspace(0, 1, n_points)
        return x

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized ZDT4 evaluation."""
        xp = get_array_namespace(X)

        f1 = X[:, 0]
        g = (
            1
            + 10 * (self.n_dim - 1)
            + xp.sum(X[:, 1:] ** 2 - 10 * xp.cos(4 * np.pi * X[:, 1:]), axis=1)
        )
        f2 = g * (1 - xp.sqrt(f1 / g))

        return xp.stack([f1, f2], axis=1)
