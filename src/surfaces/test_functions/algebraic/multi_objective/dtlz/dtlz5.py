# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""DTLZ5 multi-objective test function.

DTLZ5 introduces a degenerate Pareto front that collapses from the
full hypersphere to a lower-dimensional curve when M > 2. This tests
whether an optimizer can correctly identify and populate a degenerate
front embedded in a higher-dimensional objective space.
"""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from .._base_multi_objective import BaseMultiObjectiveTestFunction


class DTLZ5(BaseMultiObjectiveTestFunction):
    r"""DTLZ5 multi-objective test function.

    DTLZ5 modifies the spherical pattern by replacing position angles
    (except the first) with dependent angles :math:`\theta_i` that
    converge to :math:`\pi/4` on the Pareto front, collapsing the
    front to a curve in the first quadrant.

    The modified angles are:

    .. math::

        \theta_1 = \frac{\pi}{2}\,x_1, \quad
        \theta_i = \frac{\pi}{4(1 + g)}(1 + 2\,g\,x_i)
        \quad i = 2, \ldots, M-1

    with :math:`g(\mathbf{x}_d) = \sum_{i=M}^{n} (x_i - 0.5)^2`.

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
    >>> from surfaces.test_functions.algebraic.multi_objective.dtlz import DTLZ5
    >>> func = DTLZ5(n_objectives=3)
    >>> result = func(np.full(func.n_dim, 0.5))
    >>> result.shape
    (3,)
    """

    name = "DTLZ5"
    n_objectives = 3
    _k = 10
    _spec = {
        "eval_cost": 1.5,
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
        x_pos = x[: M - 1]
        x_dist = x[M - 1 :]

        g = np.sum((x_dist - 0.5) ** 2)
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
        """Degenerate front: full quarter-circle for M=2, curve for M>2."""
        M = self.n_objectives
        if M == 2:
            theta = np.linspace(0, np.pi / 2, n_points)
            return np.column_stack([np.cos(theta), np.sin(theta)])

        # On the Pareto front g=0, so theta[i>=1] = pi/4
        # Only theta[0] = x_pos[0] * pi/2 varies freely
        t0 = np.linspace(0, np.pi / 2, n_points)
        fixed_angle = np.pi / 4

        f = np.zeros((n_points, M))
        # Build from the spherical formula with theta
        cos_fixed = np.cos(fixed_angle)
        sin_fixed = np.sin(fixed_angle)

        # f[0] = cos(t0) * cos(pi/4)^(M-2)
        f[:, 0] = np.cos(t0) * cos_fixed ** (M - 2)
        # f[i] for i=1..M-2: cos(t0) * cos(pi/4)^(M-2-i) * sin(pi/4)
        for i in range(1, M - 1):
            f[:, i] = np.cos(t0) * cos_fixed ** (M - 2 - i) * sin_fixed
        # f[M-1] = sin(t0)
        f[:, M - 1] = np.sin(t0)

        return f

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
        """Vectorized DTLZ5 evaluation."""
        xp = get_array_namespace(X)
        M = self.n_objectives
        x_pos = X[:, : M - 1]
        x_dist = X[:, M - 1 :]

        g = xp.sum((x_dist - 0.5) ** 2, axis=1)
        one_plus_g = 1 + g

        # Build theta array: theta[:,0] = x_pos[:,0]*pi/2,
        # theta[:,i] = pi/(4*(1+g)) * (1 + 2*g*x_pos[:,i]) for i>=1
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
