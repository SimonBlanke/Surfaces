# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .mathematical import (
    AckleyFunction,
    BealeFunction,
    BoothFunction,
    BukinFunctionN6,
    CrossInTrayFunction,
    DropWaveFunction,
    EasomFunction,
    EggholderFunction,
    GoldsteinPriceFunction,
    GramacyAndLeeFunction,
    GriewankFunction,
    HimmelblausFunction,
    HölderTableFunction,
    LangermannFunction,
    LeviFunctionN13,
    MatyasFunction,
    McCormickFunction,
    RastriginFunction,
    RosenbrockFunction,
    SchafferFunctionN2,
    SimionescuFunction,
    SphereFunction,
    StyblinskiTangFunction,
    ThreeHumpCamelFunction,
)
from .machine_learning import (
    KNeighborsClassifierFunction,
    GradientBoostingRegressorFunction,
    KNeighborsRegressorFunction,
)


__all__ = [
    "AckleyFunction",
    "BealeFunction",
    "BoothFunction",
    "BukinFunctionN6",
    "CrossInTrayFunction",
    "DropWaveFunction",
    "EasomFunction",
    "EggholderFunction",
    "GoldsteinPriceFunction",
    "GramacyAndLeeFunction",
    "GriewankFunction",
    "HimmelblausFunction",
    "HölderTableFunction",
    "LangermannFunction",
    "LeviFunctionN13",
    "MatyasFunction",
    "McCormickFunction",
    "RastriginFunction",
    "RosenbrockFunction",
    "SchafferFunctionN2",
    "SimionescuFunction",
    "SphereFunction",
    "StyblinskiTangFunction",
    "ThreeHumpCamelFunction",
    "KNeighborsClassifierFunction",
    "GradientBoostingRegressorFunction",
    "KNeighborsRegressorFunction",
]

mathematical_functions = [
    AckleyFunction,
    BealeFunction,
    BoothFunction,
    BukinFunctionN6,
    CrossInTrayFunction,
    DropWaveFunction,
    EasomFunction,
    EggholderFunction,
    GoldsteinPriceFunction,
    GramacyAndLeeFunction,
    GriewankFunction,
    HimmelblausFunction,
    HölderTableFunction,
    LangermannFunction,
    LeviFunctionN13,
    MatyasFunction,
    McCormickFunction,
    RastriginFunction,
    RosenbrockFunction,
    SchafferFunctionN2,
    SimionescuFunction,
    SphereFunction,
    StyblinskiTangFunction,
    ThreeHumpCamelFunction,
]


mathematical_functions_1d = [
    GramacyAndLeeFunction,
]


mathematical_functions_2d = [
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
    RosenbrockFunction,
    SchafferFunctionN2,
    SimionescuFunction,
    ThreeHumpCamelFunction,
]


mathematical_functions_nd = [
    GriewankFunction,
    RastriginFunction,
    SphereFunction,
    StyblinskiTangFunction,
]

machine_learning_functions = [
    KNeighborsClassifierFunction,
    GradientBoostingRegressorFunction,
    KNeighborsRegressorFunction,
]
