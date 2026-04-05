# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""WFG3 multi-objective test function.

WFG3 is the only WFG problem with a degenerate Pareto front. By setting
the degeneracy vector A to [1, 0, 0, ...], the front collapses from an
(M-1)-dimensional surface to a one-dimensional curve, testing whether
optimizers can handle degenerate geometries.
"""

import numpy as np

from ._base_wfg import BaseWFGFunction
from ._transformations import linear, r_nonsep, s_linear


class WFG3(BaseWFGFunction):
    r"""WFG3 multi-objective test function.

    WFG3 combines non-separable distance parameters with a linear,
    degenerate Pareto front. The degeneracy vector :math:`A = [1, 0, \ldots, 0]`
    causes all but the first position group to collapse, producing a
    one-dimensional front embedded in :math:`M`-dimensional objective space.

    Parameters
    ----------
    n_objectives : int, default=3
        Number of objectives :math:`M`.
    k : int, optional
        Number of position-related parameters. Defaults to
        ``2 * (n_objectives - 1)``.
    n_dist : int, default=20
        Number of distance-related parameters. Must be even for WFG3.
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
    >>> from surfaces.test_functions.algebraic.multi_objective.wfg import WFG3
    >>> func = WFG3(n_objectives=2)
    >>> result = func(np.zeros(func.n_dim))
    >>> result.shape
    (2,)
    """

    name = "WFG3"
    n_objectives = 3
    _spec = {
        "eval_cost": 2.0,
        "continuous": True,
        "differentiable": False,
        "scalable": True,
    }

    def __init__(self, n_objectives=3, k=None, n_dist=20, **kwargs):
        if n_dist % 2 != 0:
            raise ValueError(f"n_dist must be even for WFG3, got n_dist={n_dist}")
        super().__init__(n_objectives=n_objectives, k=k, n_dist=n_dist, **kwargs)
        # Degenerate front: only the first A entry is 1
        self._A = np.zeros(n_objectives - 1)
        self._A[0] = 1.0

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
        for m in range(M):
            f[m] = x_M + self._S[m] * linear(x_head, M, m)
        return f

    def _pareto_front(self, n_points):
        """Degenerate linear Pareto front (one-dimensional curve)."""
        M = self.n_objectives
        if M == 2:
            t = np.linspace(0, 1, n_points)
            h = np.column_stack([t, 1 - t])
        else:
            t = np.linspace(0, 1, n_points)
            h = np.zeros((n_points, M))
            h[:, 0] = t
            # Degenerate: only x[0] varies, rest are 0.5
            for i in range(n_points):
                x_head = np.full(M - 1, 0.5)
                x_head[0] = t[i]
                for m in range(M):
                    h[i, m] = linear(x_head, M, m)
        return self._S * h
