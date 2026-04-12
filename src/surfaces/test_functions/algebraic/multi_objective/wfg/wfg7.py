# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""WFG7 multi-objective test function.

WFG7 introduces parameter-dependent bias on position parameters. Each
position parameter's bias depends on the mean of all subsequent
parameters, creating an ordering dependency where the optimal value of
earlier variables depends on later ones.
"""

import numpy as np

from ._base_wfg import BaseWFGFunction
from ._transformations import b_param, concave, r_sum, s_linear


class WFG7(BaseWFGFunction):
    r"""WFG7 multi-objective test function.

    WFG7 applies parameter-dependent bias to position parameters: each
    position variable is biased by a function of the weighted mean of
    all subsequent variables. Distance parameters undergo a simple
    linear shift. The reduction uses separable weighted sums.

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
    >>> from surfaces.test_functions.algebraic.multi_objective.wfg import WFG7
    >>> func = WFG7(n_objectives=2)
    >>> result = func(np.zeros(func.n_dim))
    >>> result.shape
    (2,)
    """

    name = "WFG7"
    n_objectives = 3
    _spec = {
        "eval_cost": 2.5,
        "continuous": True,
        "differentiable": True,
        "scalable": True,
    }

    def _transform(self, y):
        n = self.n_dim
        k = self._k
        M = self.n_objectives

        t1 = np.copy(y)
        for i in range(k):
            w = np.ones(n - i - 1)
            u = r_sum(y[i + 1 :], w)
            t1[i] = b_param(y[i], u, 0.98 / 49.98, 0.02, 50)

        t2 = np.copy(t1)
        for i in range(k, n):
            t2[i] = s_linear(t1[i], 0.35)

        return self._reduce(t2, k, M)

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
