# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tension/compression spring design optimization problem."""

from typing import Any, Dict, List

import numpy as np

from ._base_engineering_function import EngineeringFunction


class TensionCompressionSpringFunction(EngineeringFunction):
    """Tension/compression spring design optimization problem.

    This mechanical engineering problem involves designing a helical
    compression spring to minimize weight while satisfying constraints
    on minimum deflection, shear stress, surge frequency, and
    geometric limits.

    Problem Description
    -------------------
    A helical compression spring must be designed to carry a given load.
    The spring is characterized by wire diameter, mean coil diameter,
    and number of active coils.

    ::

              |<-- D -->|
              .----.----.
             /    /    /|
            /    /    / |
           |    |    |  |  <- d (wire diameter)
            \\    \\    \\ |
             \\    \\    \\|
              '----'----'
                  .
                  .    N active coils
                  .
              .----.----.
             /    /    /
            /    /    /

    The objective is to minimize the spring weight, which is proportional
    to the wire length (N * pi * D * d^2).

    Design Variables
    ----------------
    d : float
        Wire diameter.
        Bounds: [0.05, 2.0] inches
    D : float
        Mean coil diameter (center of wire to center).
        Bounds: [0.25, 1.3] inches
    N : float
        Number of active coils.
        Bounds: [2.0, 15.0]

    Objective Function
    ------------------
    Minimize spring weight:

    .. math::

        f(d, D, N) = (N + 2) D d^2

    This is proportional to the total wire volume (and thus weight).

    Constraints
    -----------
    1. Minimum deflection constraint
    2. Shear stress constraint
    3. Surge frequency constraint
    4. Outer diameter constraint (D + d <= D_max)

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds.
    penalty_coefficient : float, default=1e6
        Penalty coefficient for constraint violations.

    Attributes
    ----------
    f_global : float
        Best known objective value: approximately 0.012665.
    x_global : ndarray
        Best known solution: [0.05169, 0.35673, 11.2885].

    Notes
    -----
    This problem has a small feasible region relative to the search space,
    making it challenging for many optimization algorithms. The optimal
    solution lies at the boundary of multiple active constraints.

    References
    ----------
    .. [1] Belegundu, A.D. (1982). "A study of mathematical programming
           methods for structural optimization." PhD Thesis, University
           of Iowa.
    .. [2] Arora, J.S. (1989). "Introduction to Optimum Design."
           McGraw-Hill, New York.

    Examples
    --------
    >>> from surfaces.test_functions.engineering import TensionCompressionSpringFunction
    >>> func = TensionCompressionSpringFunction()
    >>> # Evaluate at a point
    >>> result = func({"d": 0.05, "D": 0.35, "N": 11.0})
    >>> # Check feasibility
    >>> func.is_feasible({"d": 0.05169, "D": 0.35673, "N": 11.2885})
    True
    """

    name = "Tension/Compression Spring Function"
    _name_ = "tension_compression_spring_function"
    __name__ = "TensionCompressionSpringFunction"

    _spec = {
        "n_dim": 3,
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    variable_names = ["d", "D", "N"]
    variable_bounds = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)]

    f_global = 0.012665
    x_global = np.array([0.05169, 0.35673, 11.2885])

    def __init__(
        self,
        objective: str = "minimize",
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks=None,
        catch_errors: bool = False,
        noise=None,
        penalty_coefficient: float = 1e6,
    ):
        super().__init__(
            objective,
            sleep,
            memory,
            collect_data,
            callbacks,
            catch_errors,
            noise,
            penalty_coefficient,
        )

    def raw_objective(self, params: Dict[str, Any]) -> float:
        """Calculate spring weight (proportional to wire volume)."""
        d = params["d"]
        D = params["D"]
        N = params["N"]

        weight = (N + 2) * D * d**2
        return weight

    def constraints(self, params: Dict[str, Any]) -> List[float]:
        """Evaluate spring design constraints."""
        d = params["d"]
        D = params["D"]
        N = params["N"]

        # Avoid division by zero
        eps = 1e-10

        # Spring index C = D/d
        C = D / (d + eps)

        # g1: Minimum deflection constraint
        # 1 - D^3*N / (71785*d^4) <= 0
        g1 = 1 - (D**3 * N) / (71785 * d**4 + eps)

        # g2: Shear stress constraint
        # (4*C^2 - C) / (12566*(C-1)*d^3) + 1/(5108*d^2) - 1 <= 0
        g2 = (4 * C**2 - C) / (12566 * (C - 1) * d**3 + eps) + 1 / (5108 * d**2 + eps) - 1

        # g3: Surge frequency constraint
        # 1 - 140.45*d / (D^2*N) <= 0
        g3 = 1 - (140.45 * d) / (D**2 * N + eps)

        # g4: Outer diameter constraint
        # (D + d) / 1.5 - 1 <= 0
        g4 = (D + d) / 1.5 - 1

        return [g1, g2, g3, g4]
