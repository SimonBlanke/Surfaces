# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .ackley_function import AckleyFunction
from .beale_function import BealeFunction
from .booth_function import BoothFunction
from .bukin_function_n6 import BukinFunctionN6
from .cross_in_tray_function import CrossInTrayFunction
from .drop_wave_function import DropWaveFunction
from .easom_function import EasomFunction
from .eggholder_function import EggholderFunction
from .goldstein_price_function import GoldsteinPriceFunction
from .gramacy_and_lee_function import GramacyAndLeeFunction
from .griewank_function import GriewankFunction
from .himmelblaus_function import HimmelblausFunction
from .hoelder_table_function import HölderTableFunction
from .langermann_function import LangermannFunction
from .levi_function_n13 import LeviFunctionN13
from .matyas_function import MatyasFunction
from .mccormick_function import McCormickFunction
from .rastrigin_function import RastriginFunction
from .rosenbrock_function import RosenbrockFunction
from .schaffer_function_n2 import SchafferFunctionN2
from .simionescu_function import SimionescuFunction
from .sphere_function import SphereFunction
from .styblinski_tang_function import StyblinskiTangFunction
from .three_hump_camel_function import ThreeHumpCamelFunction

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
