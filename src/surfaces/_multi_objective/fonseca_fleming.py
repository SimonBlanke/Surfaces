# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Fonseca-Fleming multi-objective test function."""

from typing import Any, Dict

import numpy as np

from ._base_multi_objective import MultiObjectiveFunction


class FonsecaFleming(MultiObjectiveFunction):
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
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    n_objectives : int
        Number of objectives (always 2).
    default_bounds : tuple
        Parameter bounds (-4.0, 4.0).

    References
    ----------
    .. [1] Fonseca, C. M., & Fleming, P. J. (1995). An overview of
       evolutionary algorithms in multiobjective optimization.
       Evolutionary computation, 3(1), 1-16.

    Examples
    --------
    >>> from surfaces.multi_objective import FonsecaFleming
    >>> func = FonsecaFleming(n_dim=3)
    >>> result = func(np.zeros(3))
    >>> result.shape
    (2,)
    """

    name = "Fonseca-Fleming"
    n_objectives = 2
    default_bounds = (-4.0, 4.0)

    _spec = {
        "continuous": True,
        "differentiable": True,
        "convex_front": False,
        "scalable": True,
    }

    def __init__(self, n_dim: int = 3, sleep: float = 0):
        super().__init__(n_dim, sleep)

    def _create_objective_function(self):
        n = self.n_dim
        offset = 1.0 / np.sqrt(n)

        def fonseca_fleming(params: Dict[str, Any]) -> np.ndarray:
            x = self._params_to_array(params)

            sum1 = np.sum((x - offset) ** 2)
            sum2 = np.sum((x + offset) ** 2)

            f1 = 1 - np.exp(-sum1)
            f2 = 1 - np.exp(-sum2)

            return np.array([f1, f2])

        self.pure_objective_function = fonseca_fleming

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        """Generate points on the theoretical Pareto front.

        The Pareto front is obtained when all xi = t for t in [-1/sqrt(n), 1/sqrt(n)].

        Parameters
        ----------
        n_points : int, default=100
            Number of points to generate.

        Returns
        -------
        np.ndarray
            Array of shape (n_points, 2).
        """
        n = self.n_dim
        offset = 1.0 / np.sqrt(n)

        # Pareto optimal when all xi = t
        t = np.linspace(-offset, offset, n_points)

        f1 = 1 - np.exp(-n * (t - offset) ** 2)
        f2 = 1 - np.exp(-n * (t + offset) ** 2)

        return np.column_stack([f1, f2])

    def pareto_set(self, n_points: int = 100) -> np.ndarray:
        """Generate points in the Pareto set.

        The Pareto set consists of points where all xi are equal,
        with xi in [-1/sqrt(n), 1/sqrt(n)].

        Parameters
        ----------
        n_points : int, default=100
            Number of points to generate.

        Returns
        -------
        np.ndarray
            Array of shape (n_points, n_dim).
        """
        n = self.n_dim
        offset = 1.0 / np.sqrt(n)

        t = np.linspace(-offset, offset, n_points)
        x = np.tile(t[:, np.newaxis], (1, n))

        return x
