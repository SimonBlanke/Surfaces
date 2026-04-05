# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""WFG4 multi-objective test function.

WFG4 is the simplest multimodal WFG problem. All parameters undergo the
same multi-modal shift, creating 30 local optima per dimension. The
concave Pareto front is the standard shape shared by WFG4 through WFG9.
"""

import numpy as np

from ._base_wfg import BaseWFGFunction
from ._transformations import concave, s_multi_modal


class WFG4(BaseWFGFunction):
    r"""WFG4 multi-objective test function.

    WFG4 applies the same multi-modal shift transformation to every
    parameter, creating a landscape with many local optima. The
    separable structure means each variable can be optimized
    independently despite the multimodality.

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
    >>> from surfaces.test_functions.algebraic.multi_objective.wfg import WFG4
    >>> func = WFG4(n_objectives=2)
    >>> result = func(np.zeros(func.n_dim))
    >>> result.shape
    (2,)
    """

    name = "WFG4"
    n_objectives = 3
    _spec = {
        "eval_cost": 2.5,
        "continuous": True,
        "differentiable": False,
        "scalable": True,
    }

    def _transform(self, y):
        n = self.n_dim
        k = self._k
        M = self.n_objectives

        t1 = np.zeros(n)
        for i in range(n):
            t1[i] = s_multi_modal(y[i], 30, 10, 0.35)

        return self._reduce(t1, k, M)

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
