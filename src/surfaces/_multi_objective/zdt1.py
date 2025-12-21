# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""ZDT1 multi-objective test function."""

from typing import Any, Dict

import numpy as np

from ._base_multi_objective import MultiObjectiveFunction


class ZDT1(MultiObjectiveFunction):
    """ZDT1 multi-objective test function.

    The ZDT1 function is a widely used benchmark for multi-objective
    optimization. It has a convex Pareto front.

    The function is defined as:

    .. math::

        f_1(x) = x_1

        f_2(x) = g(x) \\cdot \\left[1 - \\sqrt{\\frac{x_1}{g(x)}}\\right]

        g(x) = 1 + \\frac{9}{n-1} \\sum_{i=2}^{n} x_i

    Parameters
    ----------
    n_dim : int, default=30
        Number of input dimensions. Must be >= 2.
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    n_objectives : int
        Number of objectives (always 2).
    default_bounds : tuple
        Parameter bounds (0.0, 1.0).

    References
    ----------
    .. [1] Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of
       multiobjective evolutionary algorithms: Empirical results.
       Evolutionary computation, 8(2), 173-195.

    Examples
    --------
    >>> from surfaces.multi_objective import ZDT1
    >>> func = ZDT1(n_dim=30)
    >>> result = func(np.zeros(30))
    >>> result.shape
    (2,)
    >>> result[0]  # f1 = x1 = 0
    0.0
    """

    name = "ZDT1"
    n_objectives = 2
    default_bounds = (0.0, 1.0)

    _spec = {
        "continuous": True,
        "differentiable": True,
        "convex_front": True,
        "scalable": True,
    }

    def __init__(self, n_dim: int = 30, sleep: float = 0):
        if n_dim < 2:
            raise ValueError(f"n_dim must be >= 2, got {n_dim}")
        super().__init__(n_dim, sleep)

    def _create_objective_function(self):
        def zdt1(params: Dict[str, Any]) -> np.ndarray:
            x = self._params_to_array(params)

            f1 = x[0]

            g = 1 + 9 * np.sum(x[1:]) / (self.n_dim - 1)
            f2 = g * (1 - np.sqrt(f1 / g))

            return np.array([f1, f2])

        self.pure_objective_function = zdt1

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        """Generate points on the theoretical Pareto front.

        The Pareto front of ZDT1 is the convex curve f2 = 1 - sqrt(f1)
        for f1 in [0, 1].

        Parameters
        ----------
        n_points : int, default=100
            Number of points to generate.

        Returns
        -------
        np.ndarray
            Array of shape (n_points, 2).
        """
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        return np.column_stack([f1, f2])

    def pareto_set(self, n_points: int = 100) -> np.ndarray:
        """Generate points in the Pareto set.

        The Pareto set of ZDT1 is x1 in [0, 1] with x2 = ... = xn = 0.

        Parameters
        ----------
        n_points : int, default=100
            Number of points to generate.

        Returns
        -------
        np.ndarray
            Array of shape (n_points, n_dim).
        """
        x = np.zeros((n_points, self.n_dim))
        x[:, 0] = np.linspace(0, 1, n_points)
        return x
