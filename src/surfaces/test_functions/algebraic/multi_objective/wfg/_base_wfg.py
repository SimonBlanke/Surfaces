# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for all WFG test functions.

Handles the shared WFG framework: input normalization to [0, 1],
the post-transform reduction from M intermediate values to objective
coordinates, and default search space construction with per-dimension
bounds z_i in [0, 2*(i+1)].

Reference: Huband, S., Hingston, P., Barone, L., & While, L. (2006).
A review of multiobjective test problems and a scalable test problem toolkit.
IEEE Transactions on Evolutionary Computation, 10(5), 477-506.
"""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike

from .._base_multi_objective import BaseMultiObjectiveTestFunction
from ._transformations import r_sum


class BaseWFGFunction(BaseMultiObjectiveTestFunction):
    """Base class for WFG test functions.

    Provides the normalization, post-transform, and objective computation
    pipeline that all WFG problems share. Subclasses implement
    ``_transform()`` and ``_compute_objectives()`` to define the specific
    transformation pipeline and Pareto front shape.

    Parameters
    ----------
    n_objectives : int, default=3
        Number of objectives :math:`M`.
    k : int, optional
        Number of position-related parameters. Defaults to
        ``2 * (n_objectives - 1)``. Must be divisible by ``(n_objectives - 1)``.
    n_dist : int, default=20
        Number of distance-related parameters.
    **kwargs
        Additional keyword arguments passed to
        :class:`BaseMultiObjectiveTestFunction`.

    Attributes
    ----------
    n_objectives : int
        Number of objectives.
    n_dim : int
        Total number of decision variables (k + n_dist).
    """

    _l_default = 20
    _spec = {
        "continuous": True,
        "differentiable": True,
        "scalable": True,
    }

    def __init__(self, n_objectives=3, k=None, n_dist=20, **kwargs):
        # Accept n_dim as override: compute n_dist from n_dim - k
        n_dim_override = kwargs.pop("n_dim", None)

        if k is None:
            k = 2 * (n_objectives - 1)
        if k % (n_objectives - 1) != 0:
            raise ValueError(
                f"k must be divisible by (n_objectives-1), got k={k}, n_objectives={n_objectives}"
            )

        if n_dim_override is not None:
            if n_dim_override <= k:
                raise ValueError(
                    f"n_dim must be > k={k} for n_objectives={n_objectives}, "
                    f"got n_dim={n_dim_override}"
                )
            n_dist = n_dim_override - k

        self._k = k
        self._n_dist = n_dist
        n_dim = k + n_dist

        super().__init__(n_dim, n_objectives=n_objectives, **kwargs)

        self._S = np.array([2.0 * (i + 1) for i in range(n_objectives)])
        # Non-degenerate by default; WFG3 overrides this
        self._A = np.ones(n_objectives - 1)

    def _default_search_space(self) -> Dict[str, Any]:
        """Per-dimension bounds: x_i in [0, 2*(i+1)]."""
        search_space = {}
        dim_size = max(int(self.default_size ** (1.0 / self.n_dim)), 2)
        for i in range(self.n_dim):
            z_max = 2.0 * (i + 1)
            step = z_max / dim_size
            search_space[f"x{i}"] = np.arange(0, z_max, step)
        return search_space

    def _normalize(self, x):
        """Normalize input to [0,1] by dividing by z_max = 2*(i+1)."""
        z_max = np.array([2.0 * (i + 1) for i in range(self.n_dim)])
        return x / z_max

    def _objective(self, params: Dict[str, Any]) -> np.ndarray:
        x_raw = self._params_to_array(params)
        y = self._normalize(x_raw)
        t = self._transform(y)
        x = self._post_transform(t)
        return self._compute_objectives(x)

    def _transform(self, y):
        """Apply transformation pipeline. Override in subclasses."""
        raise NotImplementedError

    def _post_transform(self, t):
        """Convert reduced parameters t (length M) to x values.

        The first M-1 values are scaled by the degeneracy vector A,
        while the last value passes through directly as the distance
        component.
        """
        M = self.n_objectives
        x = np.zeros(M)
        for i in range(M - 1):
            x[i] = max(t[M - 1], self._A[i]) * (t[i] - 0.5) + 0.5
        x[M - 1] = t[M - 1]
        return x

    def _compute_objectives(self, x):
        """Compute objectives from x using shape functions. Override in subclasses."""
        raise NotImplementedError

    def _reduce(self, t_vec, k, M):
        """Standard WFG reduction with uniform weights.

        Groups the first k parameters into M-1 equal groups for position,
        and the remaining parameters into one group for distance.
        """
        result = np.zeros(M)
        group_size = k // (M - 1)
        for i in range(M - 1):
            start = i * group_size
            end = start + group_size
            w = np.ones(group_size)
            result[i] = r_sum(t_vec[start:end], w)
        w = np.ones(len(t_vec) - k)
        result[M - 1] = r_sum(t_vec[k:], w)
        return result

    def _reduce_weighted(self, t_vec, w, k, M):
        """WFG reduction with non-uniform weights (used by WFG1)."""
        result = np.zeros(M)
        group_size = k // (M - 1)
        for i in range(M - 1):
            start = i * group_size
            end = start + group_size
            result[i] = r_sum(t_vec[start:end], w[start:end])
        result[M - 1] = r_sum(t_vec[k:], w[k:])
        return result

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Loop-based batch evaluation.

        WFG transformations involve per-element conditional logic and
        non-separable reductions that resist straightforward vectorization.
        """
        n_points = X.shape[0]
        results = np.zeros((n_points, self.n_objectives))
        for i in range(n_points):
            params = {f"x{j}": float(X[i, j]) for j in range(self.n_dim)}
            results[i] = self._objective(params)
        return results

    def _pareto_set(self, n_points):
        """WFG Pareto set.

        Position parameters vary to trace the front; distance parameters
        are set to the values that make the distance component zero.
        For most WFG problems with s_linear(A=0.35), the optimal
        normalized position parameter value is 0.35.
        """
        M = self.n_objectives
        x = np.zeros((n_points, self.n_dim))

        if M == 2:
            t = np.linspace(0, 1, n_points)
            for i in range(self._k):
                z_max = 2.0 * (i + 1)
                x[:, i] = t * z_max
        else:
            rng = np.random.default_rng(42)
            for i in range(self._k):
                z_max = 2.0 * (i + 1)
                x[:, i] = rng.uniform(0, z_max, n_points)
        return x
