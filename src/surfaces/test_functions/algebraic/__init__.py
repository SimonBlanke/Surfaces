# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Algebraic test functions with closed-form analytical expressions.

This module contains three categories of algebraic functions:
- standard: Classic benchmark functions (Sphere, Rastrigin, Ackley, etc.)
- constrained: Engineering design problems with constraints (WeldedBeam, etc.)
- multi_objective: Multi-objective optimization problems (ZDT, Kursawe, etc.)
"""

from ._base_algebraic_function import AlgebraicFunction, MathematicalFunction

# Constrained functions (engineering design problems)
from .constrained import (
    CantileverBeamFunction,
    EngineeringFunction,
    PressureVesselFunction,
    TensionCompressionSpringFunction,
    ThreeBarTrussFunction,
    WeldedBeamFunction,
    constrained_functions,
)

# Multi-objective functions
from .multi_objective import (
    ZDT1,
    FonsecaFleming,
    Kursawe,
    MultiObjectiveFunction,
    multi_objective_functions,
)

# Standard functions (1D, 2D, ND)
from .standard import (
    # 2D
    AckleyFunction,
    BealeFunction,
    BoothFunction,
    BukinFunctionN6,
    CrossInTrayFunction,
    # 1D
    DampedSineFunction,
    DropWaveFunction,
    EasomFunction,
    EggholderFunction,
    ForresterFunction,
    GoldsteinPriceFunction,
    GramacyAndLeeFunction,
    # ND
    GriewankFunction,
    HimmelblausFunction,
    HölderTableFunction,
    LangermannFunction,
    LeviFunctionN13,
    MatyasFunction,
    McCormickFunction,
    QuadraticExponentialFunction,
    RastriginFunction,
    RosenbrockFunction,
    SchafferFunctionN2,
    SimionescuFunction,
    SineProductFunction,
    SphereFunction,
    StyblinskiTangFunction,
    ThreeHumpCamelFunction,
    standard_functions,
    standard_functions_1d,
    standard_functions_2d,
    standard_functions_nd,
)

__all__ = [
    # Base classes
    "AlgebraicFunction",
    "MathematicalFunction",
    "EngineeringFunction",
    "MultiObjectiveFunction",
    # Standard 1D
    "DampedSineFunction",
    "ForresterFunction",
    "GramacyAndLeeFunction",
    "QuadraticExponentialFunction",
    "SineProductFunction",
    # Standard 2D
    "AckleyFunction",
    "BealeFunction",
    "BoothFunction",
    "BukinFunctionN6",
    "CrossInTrayFunction",
    "DropWaveFunction",
    "EasomFunction",
    "EggholderFunction",
    "GoldsteinPriceFunction",
    "HimmelblausFunction",
    "HölderTableFunction",
    "LangermannFunction",
    "LeviFunctionN13",
    "MatyasFunction",
    "McCormickFunction",
    "SchafferFunctionN2",
    "SimionescuFunction",
    "ThreeHumpCamelFunction",
    # Standard ND
    "GriewankFunction",
    "RastriginFunction",
    "RosenbrockFunction",
    "SphereFunction",
    "StyblinskiTangFunction",
    # Constrained
    "CantileverBeamFunction",
    "PressureVesselFunction",
    "TensionCompressionSpringFunction",
    "ThreeBarTrussFunction",
    "WeldedBeamFunction",
    # Multi-objective
    "FonsecaFleming",
    "Kursawe",
    "ZDT1",
    # Function lists
    "algebraic_functions",
    "standard_functions",
    "constrained_functions",
    "multi_objective_functions",
]

# Combined list of standard algebraic functions (same as before restructuring)
# Note: constrained_functions are separate because they use different variable naming
algebraic_functions = standard_functions

# Backwards compatibility aliases
mathematical_functions = algebraic_functions
algebraic_functions_1d = standard_functions_1d
algebraic_functions_2d = standard_functions_2d
algebraic_functions_nd = standard_functions_nd
mathematical_functions_1d = standard_functions_1d
mathematical_functions_2d = standard_functions_2d
mathematical_functions_nd = standard_functions_nd
