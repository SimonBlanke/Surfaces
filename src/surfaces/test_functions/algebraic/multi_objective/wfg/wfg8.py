# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""WFG8 multi-objective test function.

WFG8 combines parameter-dependent bias on distance parameters with
non-separable reduction. Each distance parameter's bias depends on the
mean of all preceding position parameters, and the reduction couples
variables within groups.
"""

import numpy as np

from ._base_wfg import BaseWFGFunction
from ._transformations import b_param, concave, r_nonsep, r_sum, s_linear


class WFG8(BaseWFGFunction):
    r"""WFG8 multi-objective test function.

    WFG8 applies parameter-dependent bias to distance parameters (each
    biased by the mean of all preceding parameters), then a linear shift,
    and finally non-separable reduction. This creates a problem where
    the optimal distance values depend on the position parameters.

    The Pareto front is concave for all objectives.

    Parameters
    ----------
    n_objectives : int, default=3
        Number of objectives :math:`M`.
    k : int, optional
        Number of position-related parameters. Defaults to
        ``2 * (n_objectives - 1)``.
    n_dist : int, default=20
        Number of distance-related parameters.
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
    >>> from surfaces.test_functions.algebraic.multi_objective.wfg import WFG8
    >>> func = WFG8(n_objectives=2)
    >>> result = func(np.zeros(func.n_dim))
    >>> result.shape
    (2,)
    """

    name = "WFG8"
    n_objectives = 3
    _spec = {
        "eval_cost": 3.0,
        "continuous": True,
        "differentiable": True,
        "scalable": True,
    }

    def _transform(self, y):
        n = self.n_dim
        k = self._k
        M = self.n_objectives
        n_dist = n - k

        t1 = np.copy(y)
        for i in range(k, n):
            w = np.ones(i)
            u = r_sum(y[:i], w)
            t1[i] = b_param(y[i], u, 0.98 / 49.98, 0.02, 50)

        t2 = np.copy(t1)
        for i in range(k, n):
            t2[i] = s_linear(t1[i], 0.35)

        result = np.zeros(M)
        group_size = k // (M - 1)
        for i in range(M - 1):
            start = i * group_size
            end = start + group_size
            result[i] = r_nonsep(t2[start:end], group_size)
        result[M - 1] = r_nonsep(t2[k:], n_dist)
        return result

    def _compute_objectives(self, x):
        M = self.n_objectives
        f = np.zeros(M)
        x_head = x[: M - 1]
        x_M = x[M - 1]
        for m in range(M):
            f[m] = x_M + self._S[m] * concave(x_head, M, m)
        return f

    def _pareto_front(self, n_points):
        """Concave Pareto front on the positive orthant."""
        M = self.n_objectives
        if M == 2:
            theta = np.linspace(0, np.pi / 2, n_points)
            h = np.column_stack([np.sin(theta), np.cos(theta)])
        else:
            rng = np.random.default_rng(42)
            raw = np.abs(rng.standard_normal((n_points, M)))
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            h = raw / norms
        return self._S * h
