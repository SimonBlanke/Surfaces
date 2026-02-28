# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Fonseca-Fleming multi-objective test function."""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_multi_objective import BaseMultiObjectiveTestFunction


class FonsecaFleming(BaseMultiObjectiveTestFunction):
    """Fonseca-Fleming multi-objective test function.

    A classic multi-objective test function with a non-convex Pareto front.

    The function is defined as:

    .. math::

        f_1(x) = 1 - \\exp\\left(-\\sum_{i=1}^{n} \\left(x_i - \\frac{1}{\\sqrt{n}}\\right)^2\\right)

        f_2(x) = 1 - \\exp\\left(-\\sum_{i=1}^{n} \\left(x_i + \\frac{1}{\\sqrt{n}}\\right)^2\\right)

    Parameters
    ----------
    n_dim : int, default=3
        Number of input dimensions.
    **kwargs
        Additional keyword arguments passed to
        :class:`BaseMultiObjectiveTestFunction` (modifiers, memory, etc.).

    Attributes
    ----------
    n_objectives : int
        Number of objectives (always 2).

    References
    ----------
    .. [1] Fonseca, C. M., & Fleming, P. J. (1995). An overview of
       evolutionary algorithms in multiobjective optimization.
       Evolutionary computation, 3(1), 1-16.

    Examples
    --------
    >>> from surfaces.test_functions.algebraic.multi_objective import FonsecaFleming
    >>> func = FonsecaFleming(n_dim=3)
    >>> result = func(np.zeros(3))
    >>> result.shape
    (2,)
    """

    name = "Fonseca-Fleming"
    _name_ = "fonseca_fleming"
    n_objectives = 2
    _spec = {
        "continuous": True,
        "differentiable": True,
        "convex_front": False,
        "scalable": True,
        "default_bounds": (-4.0, 4.0),
    }

    def __init__(self, n_dim: int = 3, **kwargs):
        super().__init__(n_dim, **kwargs)

    def _objective(self, params: Dict[str, Any]) -> np.ndarray:
        x = self._params_to_array(params)
        n = self.n_dim
        offset = 1.0 / np.sqrt(n)

        sum1 = np.sum((x - offset) ** 2)
        sum2 = np.sum((x + offset) ** 2)

        f1 = 1 - np.exp(-sum1)
        f2 = 1 - np.exp(-sum2)

        return np.array([f1, f2])

    def _pareto_set(self, n_points: int) -> np.ndarray:
        """Pareto set: all xi equal, in [-1/sqrt(n), 1/sqrt(n)]."""
        n = self.n_dim
        offset = 1.0 / np.sqrt(n)

        t = np.linspace(-offset, offset, n_points)
        x = np.tile(t[:, np.newaxis], (1, n))

        return x

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized Fonseca-Fleming evaluation.

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
        n = self.n_dim
        offset = 1.0 / xp.sqrt(float(n))

        # sum1 = sum((x - offset)^2)
        sum1 = xp.sum((X - offset) ** 2, axis=1)
        # sum2 = sum((x + offset)^2)
        sum2 = xp.sum((X + offset) ** 2, axis=1)

        f1 = 1 - xp.exp(-sum1)
        f2 = 1 - xp.exp(-sum2)

        return xp.stack([f1, f2], axis=1)
