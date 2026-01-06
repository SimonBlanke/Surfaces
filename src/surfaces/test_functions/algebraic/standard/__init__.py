# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Standard algebraic test functions with closed-form analytical expressions."""

# 1D functions
from .test_functions_1d import (
    DampedSineFunction,
    ForresterFunction,
    GramacyAndLeeFunction,
    QuadraticExponentialFunction,
    SineProductFunction,
)

# 2D functions
from .test_functions_2d import (
    AckleyFunction,
    BealeFunction,
    BoothFunction,
    BukinFunctionN6,
    CrossInTrayFunction,
    DropWaveFunction,
    EasomFunction,
    EggholderFunction,
    GoldsteinPriceFunction,
    HimmelblausFunction,
    HölderTableFunction,
    LangermannFunction,
    LeviFunctionN13,
    MatyasFunction,
    McCormickFunction,
    SchafferFunctionN2,
    SimionescuFunction,
    ThreeHumpCamelFunction,
)

# ND functions
from .test_functions_nd import (
    GriewankFunction,
    RastriginFunction,
    RosenbrockFunction,
    SphereFunction,
    StyblinskiTangFunction,
)

__all__ = [
    # 1D
    "DampedSineFunction",
    "ForresterFunction",
    "GramacyAndLeeFunction",
    "QuadraticExponentialFunction",
    "SineProductFunction",
    # 2D
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
    # ND
    "GriewankFunction",
    "RastriginFunction",
    "RosenbrockFunction",
    "SphereFunction",
    "StyblinskiTangFunction",
]

standard_functions = [
    # 1D
    DampedSineFunction,
    ForresterFunction,
    GramacyAndLeeFunction,
    QuadraticExponentialFunction,
    SineProductFunction,
    # 2D
    AckleyFunction,
    BealeFunction,
    BoothFunction,
    BukinFunctionN6,
    CrossInTrayFunction,
    DropWaveFunction,
    EasomFunction,
    EggholderFunction,
    GoldsteinPriceFunction,
    HimmelblausFunction,
    HölderTableFunction,
    LangermannFunction,
    LeviFunctionN13,
    MatyasFunction,
    McCormickFunction,
    SchafferFunctionN2,
    SimionescuFunction,
    ThreeHumpCamelFunction,
    # ND
    GriewankFunction,
    RastriginFunction,
    RosenbrockFunction,
    SphereFunction,
    StyblinskiTangFunction,
]

standard_functions_1d = [
    DampedSineFunction,
    ForresterFunction,
    GramacyAndLeeFunction,
    QuadraticExponentialFunction,
    SineProductFunction,
]

standard_functions_2d = [
    AckleyFunction,
    BealeFunction,
    BoothFunction,
    BukinFunctionN6,
    CrossInTrayFunction,
    DropWaveFunction,
    EasomFunction,
    EggholderFunction,
    GoldsteinPriceFunction,
    HimmelblausFunction,
    HölderTableFunction,
    LangermannFunction,
    LeviFunctionN13,
    MatyasFunction,
    McCormickFunction,
    SchafferFunctionN2,
    SimionescuFunction,
    ThreeHumpCamelFunction,
]

standard_functions_nd = [
    GriewankFunction,
    RastriginFunction,
    RosenbrockFunction,
    SphereFunction,
    StyblinskiTangFunction,
]
