# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Kursawe multi-objective test function."""

from typing import Any, Dict

import numpy as np

from ._base_multi_objective import MultiObjectiveFunction


class Kursawe(MultiObjectiveFunction):
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
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    n_objectives : int
        Number of objectives (always 2).
    default_bounds : tuple
        Parameter bounds (-5.0, 5.0).

    References
    ----------
    .. [1] Kursawe, F. (1991). A variant of evolution strategies for
       vector optimization. In Parallel Problem Solving from Nature
       (pp. 193-197). Springer.

    Examples
    --------
    >>> from surfaces.multi_objective import Kursawe
    >>> func = Kursawe(n_dim=3)
    >>> result = func(np.zeros(3))
    >>> result.shape
    (2,)
    """

    name = "Kursawe"
    n_objectives = 2
    default_bounds = (-5.0, 5.0)

    _spec = {
        "continuous": True,
        "differentiable": False,  # |x|^0.8 is not differentiable at 0
        "convex_front": False,
        "disconnected_front": True,
        "scalable": True,
    }

    def __init__(self, n_dim: int = 3, sleep: float = 0):
        if n_dim < 2:
            raise ValueError(f"n_dim must be >= 2, got {n_dim}")
        super().__init__(n_dim, sleep)

    def _create_objective_function(self):
        def kursawe(params: Dict[str, Any]) -> np.ndarray:
            x = self._params_to_array(params)

            # f1: sum of exponential terms
            f1 = 0.0
            for i in range(self.n_dim - 1):
                f1 += -10 * np.exp(-0.2 * np.sqrt(x[i] ** 2 + x[i + 1] ** 2))

            # f2: sum of power and sine terms
            f2 = np.sum(np.abs(x) ** 0.8 + 5 * np.sin(x) ** 3)

            return np.array([f1, f2])

        self.pure_objective_function = kursawe

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        """Generate approximate points on the Pareto front.

        The Kursawe function has a complex, disconnected Pareto front
        that cannot be expressed analytically. This method generates
        an approximation by sampling the Pareto set.

        Parameters
        ----------
        n_points : int, default=100
            Number of points to generate.

        Returns
        -------
        np.ndarray
            Array of shape (n_points, 2).
        """
        # Sample from the approximate Pareto set and evaluate
        x_samples = self.pareto_set(n_points)
        front = np.zeros((n_points, 2))

        for i, x in enumerate(x_samples):
            params = {f"x{j}": x[j] for j in range(self.n_dim)}
            front[i] = self.pure_objective_function(params)

        return front

    def pareto_set(self, n_points: int = 100) -> np.ndarray:
        """Generate approximate points in the Pareto set.

        The Pareto set of Kursawe is approximately obtained when
        all variables are equal. This is a rough approximation.

        Parameters
        ----------
        n_points : int, default=100
            Number of points to generate.

        Returns
        -------
        np.ndarray
            Array of shape (n_points, n_dim).
        """
        # Approximate Pareto set: all variables equal, in range roughly [-1.5, 1.5]
        t = np.linspace(-1.5, 1.5, n_points)
        x = np.tile(t[:, np.newaxis], (1, self.n_dim))
        return x
