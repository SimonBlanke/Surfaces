# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""DTLZ7 multi-objective test function.

DTLZ7 produces a disconnected set of Pareto-optimal regions in
objective space. The first M-1 objectives are simply the position
parameters, while the last objective is a nonlinear function that
creates :math:`2^{M-1}` disconnected segments on the Pareto front.
"""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from .._base_multi_objective import BaseMultiObjectiveTestFunction


class DTLZ7(BaseMultiObjectiveTestFunction):
    r"""DTLZ7 multi-objective test function.

    DTLZ7 is unique among the DTLZ family in that its Pareto front
    consists of :math:`2^{M-1}` disconnected regions. The first
    :math:`M-1` objectives are identity mappings of the position
    parameters, while the last objective creates the disconnected
    geometry through a sinusoidal term.

    The objectives are:

    .. math::

        f_i(\mathbf{x}) = x_i \quad i = 1, \ldots, M-1

        f_M(\mathbf{x}) = (1 + g(\mathbf{x}_d)) \cdot h(f_1, \ldots, f_{M-1}, g)

    where:

    .. math::

        g(\mathbf{x}_d) = 1 + \frac{9}{k} \sum_{i=M}^{n} x_i

        h = M - \sum_{i=1}^{M-1} \frac{f_i}{1+g}
            \left(1 + \sin(3\pi f_i)\right)

    Parameters
    ----------
    n_objectives : int, default=3
        Number of objectives :math:`M`.
    n_dim : int, optional
        Total number of decision variables. Defaults to
        ``(n_objectives - 1) + 20``.
    **kwargs
        Additional keyword arguments passed to
        :class:`BaseMultiObjectiveTestFunction`.

    Attributes
    ----------
    n_objectives : int
        Number of objectives.

    References
    ----------
    .. [1] Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005).
       Scalable test problems for evolutionary multiobjective optimization.
       In Evolutionary Multiobjective Optimization (pp. 105-145). Springer.

    Examples
    --------
    >>> from surfaces.test_functions.algebraic.multi_objective.dtlz import DTLZ7
    >>> func = DTLZ7(n_objectives=3)
    >>> result = func(np.full(func.n_dim, 0.5))
    >>> result.shape
    (3,)
    """

    name = "DTLZ7"
    n_objectives = 3
    _k = 20
    _spec = {
        "eval_cost": 1.8,
        "continuous": True,
        "differentiable": True,
        "scalable": True,
        "default_bounds": (0.0, 1.0),
    }

    def __init__(self, n_objectives: int = 3, n_dim: int = None, **kwargs):
        if n_dim is None:
            n_dim = (n_objectives - 1) + self._k
        if n_dim < n_objectives:
            raise ValueError(
                f"n_dim must be >= n_objectives, got n_dim={n_dim}, n_objectives={n_objectives}"
            )
        super().__init__(n_dim, n_objectives=n_objectives, **kwargs)

    def _objective(self, params: Dict[str, Any]) -> np.ndarray:
        x = self._params_to_array(params)
        M = self.n_objectives
        x_dist = x[M - 1 :]

        k = len(x_dist)
        g = 1 + (9 / k) * np.sum(x_dist)

        f = np.zeros(M)
        for i in range(M - 1):
            f[i] = x[i]

        h = M - np.sum(f[: M - 1] / (1 + g) * (1 + np.sin(3 * np.pi * f[: M - 1])))
        f[M - 1] = (1 + g) * h

        return f

    def _pareto_front(self, n_points: int) -> np.ndarray:
        """Disconnected Pareto front regions with g=1 (x_dist=0)."""
        M = self.n_objectives
        # With x_dist=0: g = 1, so (1+g) = 2
        g_opt = 1.0
        one_plus_g = 1 + g_opt

        rng = np.random.default_rng(42)
        if M == 2:
            f_pos = np.linspace(0, 1, n_points).reshape(-1, 1)
        else:
            f_pos = rng.uniform(0, 1, (n_points, M - 1))

        h = M - np.sum(
            f_pos / one_plus_g * (1 + np.sin(3 * np.pi * f_pos)),
            axis=1,
        )
        f_last = one_plus_g * h

        return np.column_stack([f_pos, f_last])

    def _pareto_set(self, n_points: int) -> np.ndarray:
        """Distance params at 0.0, position params vary in [0, 1]."""
        M = self.n_objectives
        x = np.zeros((n_points, self.n_dim))
        # DTLZ7 optimum: x_dist = 0
        if M == 2:
            x[:, 0] = np.linspace(0, 1, n_points)
        else:
            rng = np.random.default_rng(42)
            x[:, : M - 1] = rng.uniform(0, 1, (n_points, M - 1))
        return x

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized DTLZ7 evaluation."""
        xp = get_array_namespace(X)
        M = self.n_objectives
        x_dist = X[:, M - 1 :]

        k = x_dist.shape[1]
        g = 1 + (9 / k) * xp.sum(x_dist, axis=1)

        f_pos = X[:, : M - 1]

        h = M - xp.sum(
            f_pos / (1 + g)[:, None] * (1 + xp.sin(3 * np.pi * f_pos)),
            axis=1,
        )
        f_last = (1 + g) * h

        return xp.concatenate([f_pos, f_last[:, None]], axis=1)
