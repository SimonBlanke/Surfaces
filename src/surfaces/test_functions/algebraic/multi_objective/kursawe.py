# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Kursawe multi-objective test function."""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ..._base_multi_objective import BaseMultiObjectiveTestFunction


class Kursawe(BaseMultiObjectiveTestFunction):
    """Kursawe multi-objective test function.

    A multi-objective test function with a non-convex, disconnected
    Pareto front consisting of three separate regions.

    The function is defined as:

    .. math::

        f_1(x) = \\sum_{i=1}^{n-1} \\left[-10 \\exp\\left(-0.2 \\sqrt{x_i^2 + x_{i+1}^2}\\right)\\right]

        f_2(x) = \\sum_{i=1}^{n} \\left[|x_i|^{0.8} + 5 \\sin^3(x_i)\\right]

    Parameters
    ----------
    n_dim : int, default=3
        Number of input dimensions. Must be >= 2.
    **kwargs
        Additional keyword arguments passed to
        :class:`BaseMultiObjectiveTestFunction` (modifiers, memory, etc.).

    Attributes
    ----------
    n_objectives : int
        Number of objectives (always 2).

    References
    ----------
    .. [1] Kursawe, F. (1991). A variant of evolution strategies for
       vector optimization. In Parallel Problem Solving from Nature
       (pp. 193-197). Springer.

    Examples
    --------
    >>> from surfaces.test_functions.algebraic.multi_objective import Kursawe
    >>> func = Kursawe(n_dim=3)
    >>> result = func(np.zeros(3))
    >>> result.shape
    (2,)
    """

    name = "Kursawe"
    n_objectives = 2
    _spec = {
        "continuous": True,
        "differentiable": False,  # |x|^0.8 is not differentiable at 0
        "convex_front": False,
        "disconnected_front": True,
        "scalable": True,
        "default_bounds": (-5.0, 5.0),
    }

    def __init__(self, n_dim: int = 3, **kwargs):
        if n_dim < 2:
            raise ValueError(f"n_dim must be >= 2, got {n_dim}")
        super().__init__(n_dim, **kwargs)

    def _objective(self, params: Dict[str, Any]) -> np.ndarray:
        x = self._params_to_array(params)

        # f1: sum of exponential terms
        f1 = 0.0
        for i in range(self.n_dim - 1):
            f1 += -10 * np.exp(-0.2 * np.sqrt(x[i] ** 2 + x[i + 1] ** 2))

        # f2: sum of power and sine terms
        f2 = np.sum(np.abs(x) ** 0.8 + 5 * np.sin(x) ** 3)

        return np.array([f1, f2])

    def _pareto_set(self, n_points: int) -> np.ndarray:
        """Approximate Pareto set: all variables equal, in [-1.5, 1.5]."""
        t = np.linspace(-1.5, 1.5, n_points)
        x = np.tile(t[:, np.newaxis], (1, self.n_dim))
        return x

    # =========================================================================
    # Batch Evaluation
    # =========================================================================

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized Kursawe evaluation.

        Parameters
        ----------
        X : ArrayLike
            Input array of shape (n_points, n_dim).

        Returns
        -------
        ArrayLike
            Output array of shape (n_points, 2).
        """
        xp = get_array_namespace(X)

        # f1 = sum_{i=1}^{n-1} [-10 * exp(-0.2 * sqrt(x_i^2 + x_{i+1}^2))]
        # Use X[:, :-1] for x_i and X[:, 1:] for x_{i+1}
        x_i = X[:, :-1]
        x_i1 = X[:, 1:]
        pairwise = -10 * xp.exp(-0.2 * xp.sqrt(x_i**2 + x_i1**2))
        f1 = xp.sum(pairwise, axis=1)

        # f2 = sum_{i=1}^{n} [|x_i|^0.8 + 5 * sin(x_i)^3]
        f2 = xp.sum(xp.abs(X) ** 0.8 + 5 * xp.sin(X) ** 3, axis=1)

        return xp.stack([f1, f2], axis=1)
