# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Pressure vessel design optimization problem."""

from typing import Any, Dict, List

import numpy as np

from ._base_engineering_function import EngineeringFunction


class PressureVesselFunction(EngineeringFunction):
    """Cylindrical pressure vessel design optimization problem.

    This problem involves designing a compressed air storage tank with
    a working pressure of 3000 psi and a minimum volume of 750 cubic feet.
    The vessel is cylindrical with hemispherical end caps.

    Problem Description
    -------------------
    The tank consists of a cylindrical shell with two hemispherical heads.
    Both the shell and heads are made from rolled steel plate, which is
    available in discrete thicknesses (multiples of 0.0625 inches).

    ::

              |<-------- L -------->|
              _______________________
           /                         \\
          (                           )  <- hemispherical
          |                           |     head (Th)
          |                           |
          |       cylindrical         |  <- shell (Ts)
          |         shell             |
          |                           |
          |           R               |  <- inner radius
          |<--------->|               |
          (                           )
           \\_______________________//

    The objective is to minimize the total cost, which includes
    material cost, forming cost, and welding cost.

    Design Variables
    ----------------
    Ts : float
        Shell thickness.
        Bounds: [0.0625, 6.1875] inches (integer multiples of 0.0625)
    Th : float
        Head thickness.
        Bounds: [0.0625, 6.1875] inches (integer multiples of 0.0625)
    R : float
        Inner radius of the vessel.
        Bounds: [10.0, 200.0] inches
    L : float
        Length of the cylindrical section (not including heads).
        Bounds: [10.0, 200.0] inches

    Objective Function
    ------------------
    Minimize total cost:

    .. math::

        f(T_s, T_h, R, L) = 0.6224 T_s R L + 1.7781 T_h R^2 + 3.1661 T_s^2 L + 19.84 T_s^2 R

    The terms represent:
    - Cylindrical shell material and welding
    - Hemispherical head material
    - Shell forming cost
    - Head-to-shell welding cost

    Constraints
    -----------
    1. Shell thickness must satisfy hoop stress requirement
    2. Head thickness must satisfy stress requirement
    3. Volume must meet minimum requirement (750 ft^3)

    Parameters
    ----------
    min_volume : float, default=750.0
        Minimum required volume in cubic feet.
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds.
    penalty_coefficient : float, default=1e6
        Penalty coefficient for constraint violations.

    Attributes
    ----------
    f_global : float
        Best known objective value: approximately 5868.76.
    x_global : ndarray
        Best known solution: [0.8125, 0.4375, 42.0984, 176.6366].

    Notes
    -----
    In the original problem formulation, Ts and Th are constrained to be
    integer multiples of 0.0625 inches (standard plate thicknesses).
    This continuous relaxation allows any value within bounds.

    References
    ----------
    .. [1] Sandgren, E. (1990). "Nonlinear integer and discrete programming
           in mechanical design optimization." Journal of Mechanical Design,
           112(2), 223-229.
    .. [2] Kannan, B.K., Kramer, S.N. (1994). "An augmented Lagrange multiplier
           based method for mixed integer discrete continuous optimization."
           Mathematical Problems in Engineering, 1, 263-275.

    Examples
    --------
    >>> from surfaces.test_functions.engineering import PressureVesselFunction
    >>> func = PressureVesselFunction()
    >>> # Evaluate at a point
    >>> result = func({"Ts": 0.8, "Th": 0.4, "R": 42.0, "L": 180.0})
    >>> # Check if design meets volume requirement
    >>> func.is_feasible({"Ts": 0.8125, "Th": 0.4375, "R": 42.0984, "L": 176.6366})
    True
    """

    name = "Pressure Vessel Function"
    _name_ = "pressure_vessel_function"
    __name__ = "PressureVesselFunction"

    _spec = {
        "n_dim": 4,
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    variable_names = ["Ts", "Th", "R", "L"]
    variable_bounds = [(0.0625, 6.1875), (0.0625, 6.1875), (10.0, 200.0), (10.0, 200.0)]

    # Function sheet attributes
    latex_formula = r"f(T_s, T_h, R, L) = 0.6224 T_s R L + 1.7781 T_h R^2 + 3.1661 T_s^2 L + 19.84 T_s^2 R"
    tagline = (
        "A cylindrical tank design problem minimizing manufacturing cost. "
        "Shell/head thickness and vessel dimensions must satisfy stress and volume constraints."
    )
    display_bounds = {"R": (10.0, 80.0), "L": (10.0, 200.0)}
    display_projection = {"dims": ("R", "L"), "fixed": {"Ts": 0.8125, "Th": 0.4375}}
    reference = "Sandgren (1990)"
    reference_url = "https://doi.org/10.1115/1.2912596"

    f_global = 5868.7649
    x_global = np.array([0.8125, 0.4375, 42.0984, 176.6366])

    def __init__(
        self,
        min_volume: float = 750.0,
        objective: str = "minimize",
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks=None,
        catch_errors: bool = False,
        noise=None,
        penalty_coefficient: float = 1e6,
    ):
        self.min_volume = min_volume
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
        """Calculate total manufacturing cost."""
        Ts = params["Ts"]
        Th = params["Th"]
        R = params["R"]
        L = params["L"]

        cost = 0.6224 * Ts * R * L + 1.7781 * Th * R**2 + 3.1661 * Ts**2 * L + 19.84 * Ts**2 * R
        return cost

    def constraints(self, params: Dict[str, Any]) -> List[float]:
        """Evaluate design constraints."""
        Ts = params["Ts"]
        Th = params["Th"]
        R = params["R"]
        L = params["L"]

        # g1: Shell thickness constraint (hoop stress)
        # -Ts + 0.0193*R <= 0
        g1 = -Ts + 0.0193 * R

        # g2: Head thickness constraint
        # -Th + 0.00954*R <= 0
        g2 = -Th + 0.00954 * R

        # g3: Volume constraint (must be at least min_volume ft^3)
        # Volume = pi*R^2*L + (4/3)*pi*R^3 (cylinder + two hemispheres)
        # Convert to cubic feet (1 ft = 12 inches)
        volume_in3 = np.pi * R**2 * L + (4 / 3) * np.pi * R**3
        volume_ft3 = volume_in3 / (12**3)  # Convert cubic inches to cubic feet
        g3 = self.min_volume - volume_ft3  # Negative when volume is sufficient

        return [g1, g2, g3]
