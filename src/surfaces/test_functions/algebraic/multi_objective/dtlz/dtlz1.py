# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""DTLZ1 multi-objective test function.

DTLZ1 uses a linear Pareto front geometry combined with a multimodal
distance function, making it challenging for optimizers to converge
to the true front rather than local attractors.
"""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from .._base_multi_objective import BaseMultiObjectiveTestFunction


class DTLZ1(BaseMultiObjectiveTestFunction):
    r"""DTLZ1 multi-objective test function.

    DTLZ1 has a linear Pareto front defined by the hyperplane
    :math:`\sum_{i=1}^{M} f_i = 0.5` in the positive orthant. The
    multimodal distance function introduces :math:`(11^k - 1)` local
    Pareto fronts that an optimizer must avoid.

    The objectives are defined as:

    .. math::

        f_1(\mathbf{x}) = \tfrac{1}{2}\,(1 + g(\mathbf{x}_d))
            \prod_{i=1}^{M-1} x_i

        f_m(\mathbf{x}) = \tfrac{1}{2}\,(1 + g(\mathbf{x}_d))
            \left(\prod_{i=1}^{M-m} x_i\right)(1 - x_{M-m+1})
            \quad m = 2, \ldots, M-1

        f_M(\mathbf{x}) = \tfrac{1}{2}\,(1 + g(\mathbf{x}_d))
            (1 - x_1)

    with the distance function:

    .. math::

        g(\mathbf{x}_d) = 100 \left[ k +
            \sum_{i=M}^{n} \left( (x_i - 0.5)^2
            - \cos(20\pi(x_i - 0.5)) \right) \right]

    where :math:`k` is the number of distance parameters and
    :math:`\mathbf{x}_d = (x_M, \ldots, x_n)`.

    Parameters
    ----------
    n_objectives : int, default=3
        Number of objectives :math:`M`.
    n_dim : int, optional
        Total number of decision variables. Defaults to
        ``(n_objectives - 1) + 5``.
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
    >>> from surfaces.test_functions.algebraic.multi_objective.dtlz import DTLZ1
    >>> func = DTLZ1(n_objectives=3)
    >>> result = func(np.full(func.n_dim, 0.5))
    >>> result.shape
    (3,)
    """

    name = "DTLZ1"
    n_objectives = 3
    _k = 5
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

        f = np.zeros(M)
        f[0] = 0.5 * (1 + g) * np.prod(x_pos)
        for i in range(1, M - 1):
            f[i] = 0.5 * (1 + g) * np.prod(x_pos[: M - 1 - i]) * (1 - x_pos[M - 1 - i])
        f[M - 1] = 0.5 * (1 + g) * (1 - x_pos[0])

        return f

    def _pareto_front(self, n_points: int) -> np.ndarray:
        """Points on the hyperplane sum(f_i) = 0.5, all f_i >= 0."""
        M = self.n_objectives
        if M == 2:
            f1 = np.linspace(0, 0.5, n_points)
            f2 = 0.5 - f1
            return np.column_stack([f1, f2])

        rng = np.random.default_rng(42)
        u = np.sort(rng.uniform(0, 1, (n_points, M - 1)), axis=1)
        diffs = np.diff(
            np.column_stack([np.zeros((n_points, 1)), u, np.ones((n_points, 1))]),
            axis=1,
        )
        return 0.5 * diffs

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
        """Vectorized DTLZ1 evaluation."""
        xp = get_array_namespace(X)
        M = self.n_objectives
        x_pos = X[:, : M - 1]
        x_dist = X[:, M - 1 :]

        k = x_dist.shape[1]
        g = 100 * (k + xp.sum((x_dist - 0.5) ** 2 - xp.cos(20 * np.pi * (x_dist - 0.5)), axis=1))

        half_1_plus_g = 0.5 * (1 + g)

        columns = []
        # f[0]: product of all position params
        columns.append(half_1_plus_g * xp.prod(x_pos, axis=1))
        # f[1] to f[M-2]: partial products with (1 - x_{M-1-i})
        for i in range(1, M - 1):
            prod_part = xp.prod(x_pos[:, : M - 1 - i], axis=1)
            columns.append(half_1_plus_g * prod_part * (1 - x_pos[:, M - 1 - i]))
        # f[M-1]: just (1 - x_pos[0])
        columns.append(half_1_plus_g * (1 - x_pos[:, 0]))

        return xp.stack(columns, axis=1)
