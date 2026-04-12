# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""DTLZ6 multi-objective test function.

DTLZ6 replaces the quadratic distance function of DTLZ5 with a
harder-to-optimize :math:`x^{0.1}` term that creates a large distance
between the initial search region and the Pareto optimal set at
:math:`x_d = 0`. The degenerate front geometry is the same as DTLZ5.
"""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from .._base_multi_objective import BaseMultiObjectiveTestFunction


class DTLZ6(BaseMultiObjectiveTestFunction):
    r"""DTLZ6 multi-objective test function.

    DTLZ6 uses the same degenerate spherical objective pattern as
    DTLZ5, but with a different distance function:

    .. math::

        g(\mathbf{x}_d) = \sum_{i=M}^{n} x_i^{0.1}

    This makes convergence significantly harder because the gradient
    near the optimum (:math:`x_d = 0`) is very steep.

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
    >>> from surfaces.test_functions.algebraic.multi_objective.dtlz import DTLZ6
    >>> func = DTLZ6(n_objectives=3)
    >>> result = func(np.full(func.n_dim, 0.5))
    >>> result.shape
    (3,)
    """

    name = "DTLZ6"
    n_objectives = 3
    _k = 10
    _spec = {
        "eval_cost": 1.5,
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

        g = np.sum(x_dist**0.1)
        one_plus_g = 1 + g

        theta = np.zeros(M - 1)
        theta[0] = x_pos[0] * np.pi / 2
        for i in range(1, M - 1):
            theta[i] = np.pi / (4 * one_plus_g) * (1 + 2 * g * x_pos[i])

        f = np.zeros(M)
        f[0] = one_plus_g * np.prod(np.cos(theta))
        for i in range(1, M - 1):
            f[i] = one_plus_g * np.prod(np.cos(theta[: M - 1 - i])) * np.sin(theta[M - 1 - i])
        f[M - 1] = one_plus_g * np.sin(theta[0])

        return f

    def _pareto_front(self, n_points: int) -> np.ndarray:
        """Degenerate front: quarter-circle for M=2, curve for M>2."""
        M = self.n_objectives
        if M == 2:
            theta = np.linspace(0, np.pi / 2, n_points)
            return np.column_stack([np.cos(theta), np.sin(theta)])

        # On the Pareto front g=0, so theta[i>=1] = pi/4
        t0 = np.linspace(0, np.pi / 2, n_points)
        fixed_angle = np.pi / 4

        f = np.zeros((n_points, M))
        cos_fixed = np.cos(fixed_angle)
        sin_fixed = np.sin(fixed_angle)

        f[:, 0] = np.cos(t0) * cos_fixed ** (M - 2)
        for i in range(1, M - 1):
            f[:, i] = np.cos(t0) * cos_fixed ** (M - 2 - i) * sin_fixed
        f[:, M - 1] = np.sin(t0)

        return f

    def _pareto_set(self, n_points: int) -> np.ndarray:
        """Distance params at 0.0, position params vary in [0, 1]."""
        M = self.n_objectives
        x = np.zeros((n_points, self.n_dim))
        # DTLZ6 optimum: x_dist = 0 (g minimized at 0)
        if M == 2:
            x[:, 0] = np.linspace(0, 1, n_points)
        else:
            rng = np.random.default_rng(42)
            x[:, : M - 1] = rng.uniform(0, 1, (n_points, M - 1))
        return x

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized DTLZ6 evaluation."""
        xp = get_array_namespace(X)
        M = self.n_objectives
        x_pos = X[:, : M - 1]
        x_dist = X[:, M - 1 :]

        g = xp.sum(x_dist**0.1, axis=1)
        one_plus_g = 1 + g

        theta_0 = x_pos[:, 0:1] * (np.pi / 2)
        if M > 2:
            scale = np.pi / (4 * one_plus_g)
            theta_rest = scale[:, None] * (1 + 2 * g[:, None] * x_pos[:, 1:])
            theta = xp.concatenate([theta_0, theta_rest], axis=1)
        else:
            theta = theta_0

        cos_theta = xp.cos(theta)
        sin_theta = xp.sin(theta)

        columns = []
        columns.append(one_plus_g * xp.prod(cos_theta, axis=1))
        for i in range(1, M - 1):
            columns.append(
                one_plus_g * xp.prod(cos_theta[:, : M - 1 - i], axis=1) * sin_theta[:, M - 1 - i]
            )
        columns.append(one_plus_g * sin_theta[:, 0])

        return xp.stack(columns, axis=1)
