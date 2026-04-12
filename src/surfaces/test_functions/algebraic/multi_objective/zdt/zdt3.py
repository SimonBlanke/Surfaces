# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""ZDT3 multi-objective test function."""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from .._base_multi_objective import BaseMultiObjectiveTestFunction

_PARETO_SEGMENTS = [
    (0.0, 0.0830),
    (0.1822, 0.2577),
    (0.4093, 0.4538),
    (0.6183, 0.6525),
    (0.8233, 0.8518),
]


def _distribute_points(n_points: int) -> list:
    """Distribute n_points across segments proportionally to their width.

    Uses largest-remainder method to guarantee the counts sum to n_points
    exactly, avoiding off-by-one issues from independent rounding.
    """
    widths = [hi - lo for lo, hi in _PARETO_SEGMENTS]
    total_width = sum(widths)

    fractions = [n_points * w / total_width for w in widths]
    floors = [max(1, int(f)) for f in fractions]
    remainders = [f - fl for f, fl in zip(fractions, floors)]

    deficit = n_points - sum(floors)
    if deficit > 0:
        for idx in np.argsort(remainders)[::-1][:deficit]:
            floors[idx] += 1

    return floors


class ZDT3(BaseMultiObjectiveTestFunction):
    """ZDT3 multi-objective test function.

    The ZDT3 function is a widely used benchmark for multi-objective
    optimization. It has a disconnected Pareto front consisting of
    five separate convex segments.

    The function is defined as:

    .. math::

        f_1(x) = x_1

        f_2(x) = g(x) \\cdot \\left[1 - \\sqrt{\\frac{x_1}{g(x)}}
                 - \\frac{x_1}{g(x)} \\sin(10 \\pi x_1)\\right]

        g(x) = 1 + \\frac{9}{n-1} \\sum_{i=2}^{n} x_i

    Parameters
    ----------
    n_dim : int, default=30
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
    >>> from surfaces.test_functions.algebraic.multi_objective.zdt import ZDT3
    >>> func = ZDT3(n_dim=30)
    >>> result = func(np.zeros(30))
    >>> result.shape
    (2,)
    >>> result[0]  # f1 = x1 = 0
    0.0
    """

    name = "ZDT3"
    n_objectives = 2
    _spec = {
        "eval_cost": 1.2,
        "continuous": True,
        "differentiable": True,
        "convex_front": False,
        "disconnected_front": True,
        "scalable": True,
        "default_bounds": (0.0, 1.0),
    }

    def __init__(self, n_dim: int = 30, **kwargs):
        if n_dim < 2:
            raise ValueError(f"n_dim must be >= 2, got {n_dim}")
        super().__init__(n_dim, **kwargs)

    def _objective(self, params: Dict[str, Any]) -> np.ndarray:
        x = self._params_to_array(params)

        f1 = x[0]

        g = 1 + 9 * np.sum(x[1:]) / (self.n_dim - 1)
        f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))

        return np.array([f1, f2])

    def _pareto_front(self, n_points: int) -> np.ndarray:
        """Pareto front: five disconnected segments where g=1."""
        counts = _distribute_points(n_points)

        segments = []
        for (lo, hi), n_seg in zip(_PARETO_SEGMENTS, counts):
            f1_seg = np.linspace(lo, hi, n_seg)
            f2_seg = 1 - np.sqrt(f1_seg) - f1_seg * np.sin(10 * np.pi * f1_seg)
            segments.append(np.column_stack([f1_seg, f2_seg]))

        return np.vstack(segments)

    def _pareto_set(self, n_points: int) -> np.ndarray:
        """Pareto set: x1 across the five segments, x2 = ... = xn = 0."""
        counts = _distribute_points(n_points)

        x1_values = []
        for (lo, hi), n_seg in zip(_PARETO_SEGMENTS, counts):
            x1_values.append(np.linspace(lo, hi, n_seg))

        x1 = np.concatenate(x1_values)

        x = np.zeros((len(x1), self.n_dim))
        x[:, 0] = x1
        return x

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized ZDT3 evaluation."""
        xp = get_array_namespace(X)

        f1 = X[:, 0]
        g = 1 + 9 * xp.sum(X[:, 1:], axis=1) / (self.n_dim - 1)
        f2 = g * (1 - xp.sqrt(f1 / g) - (f1 / g) * xp.sin(10 * np.pi * f1))

        return xp.stack([f1, f2], axis=1)
