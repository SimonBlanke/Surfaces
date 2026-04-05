# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""WFG1 multi-objective test function.

WFG1 is a separable, unimodal problem with a biased landscape. The flat
region bias on position parameters and polynomial bias on distance
parameters create a challenging landscape where naive optimizers converge
slowly. The Pareto front combines convex and mixed shapes.
"""

import numpy as np

from ._base_wfg import BaseWFGFunction
from ._transformations import (
    b_flat,
    b_poly,
    convex,
    mixed,
    s_linear,
)


class WFG1(BaseWFGFunction):
    r"""WFG1 multi-objective test function.

    WFG1 features a separable, biased landscape with a convex-mixed
    Pareto front. The transformation pipeline applies a linear shift to
    distance parameters, a flat-region bias to position parameters,
    polynomial bias to distance parameters, and a final polynomial bias
    to all parameters.

    The Pareto front geometry is convex for the first :math:`M-1`
    objectives and uses a mixed convex/concave shape for the last.

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
    >>> from surfaces.test_functions.algebraic.multi_objective.wfg import WFG1
    >>> func = WFG1(n_objectives=2)
    >>> result = func(np.zeros(func.n_dim))
    >>> result.shape
    (2,)
    """

    name = "WFG1"
    n_objectives = 3
    _spec = {
        "eval_cost": 2.0,
        "continuous": True,
        "differentiable": True,
        "scalable": True,
    }

    def _transform(self, y):
        n = self.n_dim
        k = self._k
        M = self.n_objectives

        t1 = np.copy(y)
        for i in range(k, n):
            t1[i] = s_linear(y[i], 0.35)

        t2 = np.copy(t1)
        for i in range(k):
            t2[i] = b_flat(t1[i], 0.8, 0.75, 0.85)
        for i in range(k, n):
            t2[i] = b_poly(t1[i], 0.02)

        t3 = np.zeros(n)
        for i in range(n):
            t3[i] = b_poly(t2[i], 50)

        w = np.array([2.0 * (i + 1) for i in range(n)])
        return self._reduce_weighted(t3, w, k, M)

    def _compute_objectives(self, x):
        M = self.n_objectives
        f = np.zeros(M)
        x_head = x[: M - 1]
        x_M = x[M - 1]
        for m in range(M - 1):
            f[m] = x_M + self._S[m] * convex(x_head, M, m)
        f[M - 1] = x_M + self._S[M - 1] * mixed(x_head, 1, 5)
        return f

    def _pareto_front(self, n_points):
        """Convex-mixed Pareto front with distance component at zero."""
        M = self.n_objectives
        if M == 2:
            t = np.linspace(0, np.pi / 2, n_points)
            h = np.column_stack(
                [
                    1 - np.cos(t),
                    (
                        1
                        - t / (np.pi / 2)
                        - np.cos(2 * np.pi * t / (np.pi / 2) + np.pi / 2) / (2 * np.pi)
                    )
                    ** 5,
                ]
            )
        else:
            rng = np.random.default_rng(42)
            x_head = rng.uniform(0, 1, (n_points, M - 1))
            h = np.zeros((n_points, M))
            for i in range(n_points):
                for m in range(M - 1):
                    h[i, m] = convex(x_head[i], M, m)
                h[i, M - 1] = mixed(x_head[i], 1, 5)
        return self._S * h
