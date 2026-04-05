# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""WFG2 multi-objective test function.

WFG2 introduces non-separability through pairwise reduction of distance
parameters. The Pareto front combines a convex shape for most objectives
with disconnected regions for the last objective, testing an optimizer's
ability to find all disconnected segments.
"""

import numpy as np

from ._base_wfg import BaseWFGFunction
from ._transformations import (
    convex,
    disconnected,
    r_nonsep,
    s_linear,
)


class WFG2(BaseWFGFunction):
    r"""WFG2 multi-objective test function.

    WFG2 features non-separable distance parameters (reduced in pairs)
    and a convex-disconnected Pareto front. The non-separable reduction
    creates coupling between distance variables while position variables
    remain independent.

    Parameters
    ----------
    n_objectives : int, default=3
        Number of objectives :math:`M`.
    k : int, optional
        Number of position-related parameters. Defaults to
        ``2 * (n_objectives - 1)``.
    n_dist : int, default=20
        Number of distance-related parameters. Must be even for WFG2.
    **kwargs
        Additional keyword arguments passed to :class:`BaseWFGFunction`.

    References
    ----------
    .. [1] Huband, S., Hingston, P., Barone, L., & While, L. (2006).
       A review of multiobjective test problems and a scalable test
       problem toolkit. IEEE Transactions on Evolutionary Computation,
       10(5), 477-506.

    Examples
    --------
    >>> from surfaces.test_functions.algebraic.multi_objective.wfg import WFG2
    >>> func = WFG2(n_objectives=2)
    >>> result = func(np.zeros(func.n_dim))
    >>> result.shape
    (2,)
    """

    name = "WFG2"
    n_objectives = 3
    _spec = {
        "eval_cost": 2.0,
        "continuous": True,
        "differentiable": False,
        "scalable": True,
    }

    def __init__(self, n_objectives=3, k=None, n_dist=20, **kwargs):
        if n_dist % 2 != 0:
            raise ValueError(f"n_dist must be even for WFG2, got n_dist={n_dist}")
        super().__init__(n_objectives=n_objectives, k=k, n_dist=n_dist, **kwargs)

    def _transform(self, y):
        n = self.n_dim
        k = self._k
        M = self.n_objectives
        n_dist = n - k

        t1 = np.copy(y)
        for i in range(k, n):
            t1[i] = s_linear(y[i], 0.35)

        half_dist = n_dist // 2
        t2 = np.zeros(k + half_dist)
        t2[:k] = t1[:k]
        for i in range(half_dist):
            t2[k + i] = r_nonsep(t1[k + 2 * i : k + 2 * (i + 1)], 2)

        w = np.ones(k + half_dist)
        return self._reduce_weighted(t2, w, k, M)

    def _compute_objectives(self, x):
        M = self.n_objectives
        f = np.zeros(M)
        x_head = x[: M - 1]
        x_M = x[M - 1]
        for m in range(M - 1):
            f[m] = x_M + self._S[m] * convex(x_head, M, m)
        f[M - 1] = x_M + self._S[M - 1] * disconnected(x_head, 1, 1, 5)
        return f

    def _pareto_front(self, n_points):
        """Convex-disconnected Pareto front."""
        M = self.n_objectives
        if M == 2:
            t = np.linspace(0, np.pi / 2, n_points)
            h = np.column_stack(
                [
                    1 - np.cos(t),
                    1
                    - np.sin(t / (np.pi / 2) * np.pi / 2) ** 1
                    * np.cos(1 * np.sin(t / (np.pi / 2)) ** 5 * np.pi) ** 2,
                ]
            )
        else:
            rng = np.random.default_rng(42)
            x_head = rng.uniform(0, 1, (n_points, M - 1))
            h = np.zeros((n_points, M))
            for i in range(n_points):
                for m in range(M - 1):
                    h[i, m] = convex(x_head[i], M, m)
                h[i, M - 1] = disconnected(x_head[i], 1, 1, 5)
        return self._S * h
