# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Engineering design optimization benchmark functions.

This module provides a collection of classic engineering design optimization
problems. These are real-world inspired benchmarks with physical meaning,
commonly used to evaluate constrained optimization algorithms.

All problems are formulated as minimization with constraints handled via
penalty methods. Each function provides:

- `raw_objective(params)`: The pure engineering objective (cost, weight, etc.)
- `constraints(params)`: Constraint function values (g <= 0 is feasible)
- `constraint_violations(params)`: Positive violation amounts
- `is_feasible(params)`: Boolean feasibility check
- `penalty(params)`: Total penalty for constraint violations

The default `__call__` method returns the penalized objective:
f(x) + penalty_coefficient * sum(max(0, g_i(x))^2)

Available Functions
-------------------
Structural Engineering:
    ThreeBarTrussFunction : 2D, 3 constraints
        Minimize weight of a symmetric truss structure.
    CantileverBeamFunction : 5D, 1 constraint
        Minimize weight of a stepped cantilever beam.

Manufacturing/Mechanical:
    WeldedBeamFunction : 4D, 5 constraints
        Minimize fabrication cost of a welded beam joint.
    PressureVesselFunction : 4D, 3 constraints
        Minimize cost of a cylindrical pressure vessel.
    TensionCompressionSpringFunction : 3D, 4 constraints
        Minimize weight of a helical compression spring.

Examples
--------
>>> from surfaces.test_functions.engineering import WeldedBeamFunction
>>> func = WeldedBeamFunction()
>>> # Evaluate with penalty
>>> cost = func({"h": 0.2, "l": 3.5, "t": 9.0, "b": 0.2})
>>> # Check raw objective without penalty
>>> raw_cost = func.raw_objective({"h": 0.2, "l": 3.5, "t": 9.0, "b": 0.2})
>>> # Check if solution is feasible
>>> func.is_feasible({"h": 0.2, "l": 3.5, "t": 9.0, "b": 0.2})

References
----------
Most problems originate from:

.. [1] Coello, C.A.C. (2000). "Use of a self-adaptive penalty approach
       for engineering optimization problems."
.. [2] Deb, K. (2000). "An efficient constraint handling method for
       genetic algorithms."
"""

from ._base_engineering_function import EngineeringFunction
from .cantilever_beam import CantileverBeamFunction
from .pressure_vessel import PressureVesselFunction
from .tension_compression_spring import TensionCompressionSpringFunction
from .three_bar_truss import ThreeBarTrussFunction
from .welded_beam import WeldedBeamFunction

__all__ = [
    "EngineeringFunction",
    "ThreeBarTrussFunction",
    "WeldedBeamFunction",
    "PressureVesselFunction",
    "TensionCompressionSpringFunction",
    "CantileverBeamFunction",
    "engineering_functions",
]

engineering_functions = [
    ThreeBarTrussFunction,
    WeldedBeamFunction,
    PressureVesselFunction,
    TensionCompressionSpringFunction,
    CantileverBeamFunction,
]
