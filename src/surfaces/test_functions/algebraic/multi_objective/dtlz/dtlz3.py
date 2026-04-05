# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""DTLZ3 multi-objective test function.

DTLZ3 combines the spherical Pareto front of DTLZ2 with the multimodal
distance function of DTLZ1. The result is a problem that is hard to
converge on (many local fronts) while the true front has a well-known
spherical shape for verification.
"""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from .._base_multi_objective import BaseMultiObjectiveTestFunction


class DTLZ3(BaseMultiObjectiveTestFunction):
    r"""DTLZ3 multi-objective test function.

    DTLZ3 shares the spherical Pareto front geometry of DTLZ2 but uses
    the highly multimodal distance function from DTLZ1. This creates
    :math:`(3^k - 1)` local Pareto fronts, all parallel to the true
    front, testing an optimizer's convergence ability.

    The objectives follow the spherical pattern:

    .. math::

        f_1(\mathbf{x}) = (1 + g(\mathbf{x}_d))
            \prod_{i=1}^{M-1} \cos\!\left(\frac{\pi}{2}\,x_i\right)

    with the multimodal distance function:

    .. math::

        g(\mathbf{x}_d) = 100 \left[ k +
            \sum_{i=M}^{n} \left( (x_i - 0.5)^2
            - \cos(20\pi(x_i - 0.5)) \right) \right]

    Parameters
    ----------
    n_objectives : int, default=3
        Number of objectives :math:`M`.
    n_dim : int, optional
        Total number of decision variables. Defaults to
        ``(n_objectives - 1) + 10``.
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
    >>> from surfaces.test_functions.algebraic.multi_objective.dtlz import DTLZ3
    >>> func = DTLZ3(n_objectives=3)
    >>> result = func(np.full(func.n_dim, 0.5))
    >>> result.shape
    (3,)
    """

    name = "DTLZ3"
    n_objectives = 3
    _k = 10
    _spec = {
        "eval_cost": 2.0,
        "continuous": True,
        "differentiable": False,
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
        x_pos = x[: M - 1]
        x_dist = x[M - 1 :]

        k = len(x_dist)
        g = 100 * (k + np.sum((x_dist - 0.5) ** 2 - np.cos(20 * np.pi * (x_dist - 0.5))))
        one_plus_g = 1 + g

        f = np.zeros(M)
        f[0] = one_plus_g * np.prod(np.cos(x_pos * np.pi / 2))
        for i in range(1, M - 1):
            f[i] = (
                one_plus_g
                * np.prod(np.cos(x_pos[: M - 1 - i] * np.pi / 2))
                * np.sin(x_pos[M - 1 - i] * np.pi / 2)
            )
        f[M - 1] = one_plus_g * np.sin(x_pos[0] * np.pi / 2)

        return f

    def _pareto_front(self, n_points: int) -> np.ndarray:
        """Points on the positive orthant of the unit hypersphere."""
        M = self.n_objectives
        if M == 2:
            theta = np.linspace(0, np.pi / 2, n_points)
            return np.column_stack([np.cos(theta), np.sin(theta)])

        rng = np.random.default_rng(42)
        raw = np.abs(rng.standard_normal((n_points, M)))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / norms

    def _pareto_set(self, n_points: int) -> np.ndarray:
        """Distance params at 0.5, position params vary in [0, 1]."""
        M = self.n_objectives
        x = np.zeros((n_points, self.n_dim))
        x[:, M - 1 :] = 0.5
        if M == 2:
            x[:, 0] = np.linspace(0, 1, n_points)
        else:
            rng = np.random.default_rng(42)
            x[:, : M - 1] = rng.uniform(0, 1, (n_points, M - 1))
        return x

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized DTLZ3 evaluation."""
        xp = get_array_namespace(X)
        M = self.n_objectives
        x_pos = X[:, : M - 1]
        x_dist = X[:, M - 1 :]

        k = x_dist.shape[1]
        g = 100 * (k + xp.sum((x_dist - 0.5) ** 2 - xp.cos(20 * np.pi * (x_dist - 0.5)), axis=1))
        one_plus_g = 1 + g

        cos_pos = xp.cos(x_pos * (np.pi / 2))
        sin_pos = xp.sin(x_pos * (np.pi / 2))

        columns = []
        columns.append(one_plus_g * xp.prod(cos_pos, axis=1))
        for i in range(1, M - 1):
            columns.append(
                one_plus_g * xp.prod(cos_pos[:, : M - 1 - i], axis=1) * sin_pos[:, M - 1 - i]
            )
        columns.append(one_plus_g * sin_pos[:, 0])

        return xp.stack(columns, axis=1)
