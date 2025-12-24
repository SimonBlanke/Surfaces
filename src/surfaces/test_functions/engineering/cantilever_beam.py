# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Cantilever beam design optimization problem."""

from typing import Any, Dict, List

import numpy as np

from ._base_engineering_function import EngineeringFunction


class CantileverBeamFunction(EngineeringFunction):
    """Cantilever beam design optimization problem.

    This structural engineering problem involves designing a stepped
    cantilever beam with minimum weight while constraining the tip
    deflection. The beam has a square cross-section that varies
    along its length in discrete steps.

    Problem Description
    -------------------
    A cantilever beam is fixed at one end and carries a vertical load
    at the free end. The beam is divided into five segments of equal
    length, each with a square cross-section of different width.
    The widths must be optimized to minimize total weight while
    keeping tip deflection below a specified limit.

    ::

        WALL
        ####|   x1    |   x2    |   x3    |   x4    |   x5    |
        ####|=========|=========|=========|=========|=========| <- P
        ####|   x1    |   x2    |   x3    |   x4    |   x5    |
        ####|_________|_________|_________|_________|_________|
             |<- l ->|

        Each segment has square cross-section with side length xi.
        Total length = 5l, where l is segment length.

    Design Variables
    ----------------
    x1 : float
        Width of segment 1 (nearest to wall).
        Bounds: [0.01, 100.0]
    x2 : float
        Width of segment 2.
        Bounds: [0.01, 100.0]
    x3 : float
        Width of segment 3.
        Bounds: [0.01, 100.0]
    x4 : float
        Width of segment 4.
        Bounds: [0.01, 100.0]
    x5 : float
        Width of segment 5 (at free end).
        Bounds: [0.01, 100.0]

    Objective Function
    ------------------
    Minimize beam weight (volume for uniform density):

    .. math::

        f(x_1, ..., x_5) = 0.0624 (x_1 + x_2 + x_3 + x_4 + x_5)

    The coefficient 0.0624 comes from segment length and density normalization.

    Constraints
    -----------
    Tip deflection must not exceed the specified limit:

    .. math::

        g: \\frac{61}{x_1^3} + \\frac{37}{x_2^3} + \\frac{19}{x_3^3} + \\frac{7}{x_4^3} + \\frac{1}{x_5^3} \\leq 1

    These coefficients arise from the structural analysis of a stepped
    cantilever beam under end loading.

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
        Best known objective value: approximately 1.3400.
    x_global : ndarray
        Best known solution: [6.0089, 5.3049, 4.5023, 3.5077, 2.1504].

    Notes
    -----
    This problem has a smooth, well-behaved landscape but the optimal
    solution requires careful balancing of material distribution along
    the beam. Segments near the fixed end (higher bending moment)
    require more material than those near the free end.

    The deflection formula comes from applying Castigliano's theorem
    to a stepped beam with square cross-section.

    References
    ----------
    .. [1] Fleury, C., Braibant, V. (1986). "Structural optimization:
           A new dual method using mixed variables." International
           Journal for Numerical Methods in Engineering, 23, 409-428.

    Examples
    --------
    >>> from surfaces.test_functions.engineering import CantileverBeamFunction
    >>> func = CantileverBeamFunction()
    >>> # Evaluate at a point
    >>> result = func({"x1": 6.0, "x2": 5.3, "x3": 4.5, "x4": 3.5, "x5": 2.2})
    >>> # Check deflection constraint
    >>> func.is_feasible({"x1": 6.0089, "x2": 5.3049, "x3": 4.5023, "x4": 3.5077, "x5": 2.1504})
    True
    """

    name = "Cantilever Beam Function"
    _name_ = "cantilever_beam_function"
    __name__ = "CantileverBeamFunction"

    _spec = {
        "n_dim": 5,
        "convex": False,
        "unimodal": True,
        "separable": True,  # Objective is separable
        "scalable": False,
    }

    variable_names = ["x1", "x2", "x3", "x4", "x5"]
    variable_bounds = [(0.01, 100.0), (0.01, 100.0), (0.01, 100.0), (0.01, 100.0), (0.01, 100.0)]

    f_global = 1.3400
    x_global = np.array([6.0089, 5.3049, 4.5023, 3.5077, 2.1504])

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
        """Calculate beam weight (total volume)."""
        x1 = params["x1"]
        x2 = params["x2"]
        x3 = params["x3"]
        x4 = params["x4"]
        x5 = params["x5"]

        weight = 0.0624 * (x1 + x2 + x3 + x4 + x5)
        return weight

    def constraints(self, params: Dict[str, Any]) -> List[float]:
        """Evaluate deflection constraint."""
        x1 = params["x1"]
        x2 = params["x2"]
        x3 = params["x3"]
        x4 = params["x4"]
        x5 = params["x5"]

        # Avoid division by zero
        eps = 1e-10

        # Deflection constraint
        # Sum of contributions from each segment
        deflection = (
            61 / (x1**3 + eps)
            + 37 / (x2**3 + eps)
            + 19 / (x3**3 + eps)
            + 7 / (x4**3 + eps)
            + 1 / (x5**3 + eps)
        )

        # g <= 0 is feasible, so deflection - 1 <= 0
        g = deflection - 1

        return [g]
