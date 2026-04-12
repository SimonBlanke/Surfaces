# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""WFG5 multi-objective test function.

WFG5 uses deceptive shift transformations on all parameters, creating a
landscape where the true global optimum is surrounded by deceptive local
optima. This is one of the hardest separable WFG problems because
gradient-based methods are easily misled.
"""

import numpy as np

from ._base_wfg import BaseWFGFunction
from ._transformations import concave, s_deceptive


class WFG5(BaseWFGFunction):
    r"""WFG5 multi-objective test function.

    WFG5 applies deceptive shift transformations to every parameter.
    The deceptive landscape has a global optimum at 0.35 for each
    normalized variable, but strong local attractors pull optimizers
    away from it.

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
    >>> from surfaces.test_functions.algebraic.multi_objective.wfg import WFG5
    >>> func = WFG5(n_objectives=2)
    >>> result = func(np.zeros(func.n_dim))
    >>> result.shape
    (2,)
    """

    name = "WFG5"
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
            t1[i] = s_deceptive(y[i], 0.35, 0.001, 0.05)

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
