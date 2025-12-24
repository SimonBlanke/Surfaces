# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Welded beam design optimization problem."""

from typing import Any, Dict, List

import numpy as np

from ._base_engineering_function import EngineeringFunction


class WeldedBeamFunction(EngineeringFunction):
    """Welded beam design optimization problem.

    This is one of the most widely used engineering benchmark problems
    in optimization literature. The goal is to design a welded beam
    for minimum fabrication cost while satisfying constraints on
    shear stress, bending stress, buckling load, and end deflection.

    Problem Description
    -------------------
    A rigid member is welded to a beam, which is attached to a wall.
    The beam must support a load P applied at the end. The weld and
    beam geometry must be optimized to minimize cost.

    ::

                    |<------ L ------->|
                    |                  |
            ========+==================+ <- beam (t x b)
            ========|                  |
             weld ->|                  |
            (h x l) |                  |
                    |                  * <- P (load)
            --------+------------------+
                WALL

    The weld has dimensions h (height) and l (length).
    The beam has dimensions t (thickness) and b (height/depth).

    Design Variables
    ----------------
    h : float
        Weld height (thickness of weld bead).
        Bounds: [0.125, 5.0] inches
    l : float
        Weld length (along the beam).
        Bounds: [0.1, 10.0] inches
    t : float
        Beam thickness (width).
        Bounds: [0.1, 10.0] inches
    b : float
        Beam height (depth).
        Bounds: [0.125, 5.0] inches

    Objective Function
    ------------------
    Minimize fabrication cost:

    .. math::

        f(h, l, t, b) = 1.10471 h^2 l + 0.04811 t b (14.0 + l)

    The first term represents weld cost (proportional to weld volume),
    and the second term represents beam material cost.

    Constraints
    -----------
    1. Shear stress in weld must not exceed allowable (tau <= tau_max)
    2. Bending stress in beam must not exceed allowable (sigma <= sigma_max)
    3. Beam thickness must not exceed weld height (h >= t)
    4. Buckling load must exceed applied load (P <= P_c)
    5. End deflection must not exceed limit (delta <= delta_max)

    Parameters
    ----------
    P : float, default=6000.0
        Applied load (lb).
    L : float, default=14.0
        Beam length from wall to load (inches).
    E : float, default=30e6
        Elastic modulus (psi).
    G : float, default=12e6
        Shear modulus (psi).
    tau_max : float, default=13600.0
        Maximum allowable shear stress (psi).
    sigma_max : float, default=30000.0
        Maximum allowable bending stress (psi).
    delta_max : float, default=0.25
        Maximum allowable deflection (inches).
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds.
    penalty_coefficient : float, default=1e6
        Penalty coefficient for constraint violations.

    Attributes
    ----------
    f_global : float
        Best known objective value: approximately 1.7248.
    x_global : ndarray
        Best known solution: [0.2057, 3.4705, 9.0366, 0.2057].

    References
    ----------
    .. [1] Ragsdell, K.M., Phillips, D.T. (1976). "Optimal design of a
           class of welded structures using geometric programming."
           Journal of Engineering for Industry, 98(3), 1021-1025.
    .. [2] Deb, K. (1991). "Optimal design of a welded beam via genetic
           algorithms." AIAA Journal, 29(11), 2013-2015.

    Examples
    --------
    >>> from surfaces.test_functions.engineering import WeldedBeamFunction
    >>> func = WeldedBeamFunction()
    >>> # Evaluate at a point
    >>> result = func({"h": 0.2, "l": 3.5, "t": 9.0, "b": 0.2})
    >>> # Check constraint violations
    >>> violations = func.constraint_violations({"h": 0.2, "l": 3.5, "t": 9.0, "b": 0.2})
    """

    name = "Welded Beam Function"
    _name_ = "welded_beam_function"
    __name__ = "WeldedBeamFunction"

    _spec = {
        "n_dim": 4,
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    variable_names = ["h", "l", "t", "b"]
    variable_bounds = [(0.125, 5.0), (0.1, 10.0), (0.1, 10.0), (0.125, 5.0)]

    f_global = 1.724852
    x_global = np.array([0.205730, 3.470489, 9.036624, 0.205730])

    def __init__(
        self,
        P: float = 6000.0,
        L: float = 14.0,
        E: float = 30e6,
        G: float = 12e6,
        tau_max: float = 13600.0,
        sigma_max: float = 30000.0,
        delta_max: float = 0.25,
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
        self.L = L
        self.E = E
        self.G = G
        self.tau_max = tau_max
        self.sigma_max = sigma_max
        self.delta_max = delta_max
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
        """Calculate fabrication cost."""
        h = params["h"]
        l = params["l"]
        t = params["t"]
        b = params["b"]

        # Weld cost + beam material cost
        cost = 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)
        return cost

    def _calculate_stresses(self, params: Dict[str, Any]) -> tuple:
        """Calculate shear and bending stresses."""
        h = params["h"]
        l = params["l"]
        t = params["t"]
        b = params["b"]
        P = self.P
        L = self.L

        # Weld geometry
        # Primary shear stress
        tau_prime = P / (np.sqrt(2) * h * l)

        # Secondary shear stress (due to moment)
        M = P * (L + l / 2)
        R = np.sqrt(l**2 / 4 + ((h + t) / 2) ** 2)
        J = 2 * (np.sqrt(2) * h * l * (l**2 / 12 + ((h + t) / 2) ** 2))

        tau_double_prime = M * R / J

        # Combined shear stress (using resultant)
        cos_theta = l / (2 * R)
        tau = np.sqrt(
            tau_prime**2 + 2 * tau_prime * tau_double_prime * cos_theta + tau_double_prime**2
        )

        # Bending stress in beam
        sigma = 6 * P * L / (t * b**2)

        return tau, sigma

    def constraints(self, params: Dict[str, Any]) -> List[float]:
        """Evaluate design constraints."""
        h = params["h"]
        l = params["l"]
        t = params["t"]
        b = params["b"]
        P = self.P
        L = self.L
        E = self.E

        tau, sigma = self._calculate_stresses(params)

        # g1: Shear stress constraint
        g1 = tau - self.tau_max

        # g2: Bending stress constraint
        g2 = sigma - self.sigma_max

        # g3: Beam thickness vs weld height
        g3 = h - t

        # g4: Buckling constraint
        # Critical buckling load
        P_c = (
            4.013
            * E
            * np.sqrt(t**2 * b**6 / 36)
            / L**2
            * (1 - t / (2 * L) * np.sqrt(E / (4 * self.G)))
        )
        g4 = P - P_c

        # g5: Deflection constraint
        delta = 4 * P * L**3 / (E * t * b**3)
        g5 = delta - self.delta_max

        return [g1, g2, g3, g4, g5]
