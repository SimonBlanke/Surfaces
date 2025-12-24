# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Three-bar truss design optimization problem."""

from typing import Any, Dict, List

import numpy as np

from ._base_engineering_function import EngineeringFunction


class ThreeBarTrussFunction(EngineeringFunction):
    """Three-bar planar truss design optimization problem.

    This classic structural engineering problem involves designing a
    three-bar planar truss to support a load P at minimum weight while
    satisfying stress constraints in each member.

    Problem Description
    -------------------
    The truss consists of three members arranged symmetrically:
    - Two diagonal members (area A1) at 45 degrees
    - One vertical member (area A2)

    The structure must support a vertical load P applied at the
    junction point. The objective is to minimize the total weight
    (proportional to total material volume) while ensuring that
    stresses in all members remain below the allowable stress.

    ::

            |------ L ------|
            *               *
             \\             /
              \\ A1     A1 /
               \\         /
                \\       /
                 \\     /
                  \\   /
                   \\ /
                    * ---- A2
                    |
                    | P (load)
                    v

    Design Variables
    ----------------
    A1 : float
        Cross-sectional area of diagonal members (dimensionless, normalized).
        Bounds: [0, 1]
    A2 : float
        Cross-sectional area of vertical member (dimensionless, normalized).
        Bounds: [0, 1]

    Objective Function
    ------------------
    Minimize weight:

    .. math::

        f(A_1, A_2) = (2\\sqrt{2} A_1 + A_2) \\cdot L

    where L is the member length (normalized to 1).

    Constraints
    -----------
    Three stress constraints ensure members don't exceed allowable stress:

    .. math::

        g_1: \\frac{\\sqrt{2} A_1 + A_2}{\\sqrt{2} A_1^2 + 2 A_1 A_2} P - \\sigma_{max} \\leq 0

        g_2: \\frac{A_2}{\\sqrt{2} A_1^2 + 2 A_1 A_2} P - \\sigma_{max} \\leq 0

        g_3: \\frac{1}{A_1 + \\sqrt{2} A_2} P - \\sigma_{max} \\leq 0

    Parameters
    ----------
    P : float, default=2.0
        Applied load magnitude.
    sigma_max : float, default=2.0
        Maximum allowable stress.
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds.
    penalty_coefficient : float, default=1e6
        Penalty coefficient for constraint violations.

    Attributes
    ----------
    f_global : float
        Best known objective value: approximately 263.896 (for default parameters).
    x_global : ndarray
        Best known solution: approximately [0.789, 0.408].

    References
    ----------
    .. [1] Coello, C.A.C. (2000). "Use of a self-adaptive penalty approach
           for engineering optimization problems." Computers in Industry,
           41(2), 113-127.

    Examples
    --------
    >>> from surfaces.test_functions.engineering import ThreeBarTrussFunction
    >>> func = ThreeBarTrussFunction()
    >>> # Evaluate at a point
    >>> result = func({"A1": 0.5, "A2": 0.5})
    >>> # Check if solution is feasible
    >>> func.is_feasible({"A1": 0.789, "A2": 0.408})
    True
    """

    name = "Three-Bar Truss Function"
    _name_ = "three_bar_truss_function"
    __name__ = "ThreeBarTrussFunction"

    _spec = {
        "n_dim": 2,
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    variable_names = ["A1", "A2"]
    variable_bounds = [(0.0, 1.0), (0.0, 1.0)]

    f_global = 263.8958434
    x_global = np.array([0.78867513, 0.40824829])

    def __init__(
        self,
        P: float = 2.0,
        sigma_max: float = 2.0,
        objective: str = "minimize",
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks=None,
        catch_errors: bool = False,
        noise=None,
        penalty_coefficient: float = 1e6,
    ):
        self.P = P
        self.sigma_max = sigma_max
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
        """Calculate weight of the truss structure."""
        A1 = params["A1"]
        A2 = params["A2"]

        # Weight proportional to total material volume
        # L = 100 (typical normalization)
        L = 100.0
        weight = L * (2 * np.sqrt(2) * A1 + A2)
        return weight

    def constraints(self, params: Dict[str, Any]) -> List[float]:
        """Evaluate stress constraints."""
        A1 = params["A1"]
        A2 = params["A2"]
        P = self.P
        sigma_max = self.sigma_max

        # Avoid division by zero
        eps = 1e-10

        # Denominator terms
        denom1 = np.sqrt(2) * A1**2 + 2 * A1 * A2 + eps
        denom2 = A1 + np.sqrt(2) * A2 + eps

        # Stress constraints (g <= 0 is feasible)
        g1 = ((np.sqrt(2) * A1 + A2) / denom1) * P - sigma_max
        g2 = (A2 / denom1) * P - sigma_max
        g3 = (1 / denom2) * P - sigma_max

        return [g1, g2, g3]
