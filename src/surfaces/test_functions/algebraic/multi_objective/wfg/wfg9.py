# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""WFG9 multi-objective test function.

WFG9 is the most complex WFG problem, combining parameter-dependent bias,
deceptive shifts on position parameters, multi-modal shifts on distance
parameters, and non-separable reduction. It exercises all transformation
types simultaneously.
"""

import numpy as np

from ._base_wfg import BaseWFGFunction
from ._transformations import (
    b_param,
    concave,
    r_nonsep,
    r_sum,
    s_deceptive,
    s_multi_modal,
)


class WFG9(BaseWFGFunction):
    r"""WFG9 multi-objective test function.

    WFG9 combines every WFG transformation type: parameter-dependent
    bias on all but the last variable, deceptive shift on position
    parameters, multi-modal shift on distance parameters, and
    non-separable reduction for all groups.

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
    >>> from surfaces.test_functions.algebraic.multi_objective.wfg import WFG9
    >>> func = WFG9(n_objectives=2)
    >>> result = func(np.zeros(func.n_dim))
    >>> result.shape
    (2,)
    """

    name = "WFG9"
    n_objectives = 3
    _spec = {
        "eval_cost": 3.0,
        "continuous": True,
        "differentiable": False,
        "scalable": True,
    }

    def _transform(self, y):
        n = self.n_dim
        k = self._k
        M = self.n_objectives
        n_dist = n - k

        # Parameter-dependent bias on all except the last variable
        t1 = np.copy(y)
        for i in range(n - 1):
            w = np.ones(n - i - 1)
            u = r_sum(y[i + 1 :], w)
            t1[i] = b_param(y[i], u, 0.98 / 49.98, 0.02, 50)

        # Deceptive shift on position params, multi-modal on distance
        t2 = np.copy(t1)
        for i in range(k):
            t2[i] = s_deceptive(t1[i], 0.35, 0.001, 0.05)
        for i in range(k, n):
            t2[i] = s_multi_modal(t1[i], 30, 95, 0.35)

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
