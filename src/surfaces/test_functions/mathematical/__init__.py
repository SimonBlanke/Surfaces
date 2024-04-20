# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .test_functions_1d.gramacy_and_lee_function import GramacyAndLeeFunction


from .test_functions_2d.ackley_function import AckleyFunction
from .test_functions_2d.beale_function import BealeFunction
from .test_functions_2d.booth_function import BoothFunction
from .test_functions_2d.bukin_function_n6 import BukinFunctionN6
from .test_functions_2d.cross_in_tray_function import CrossInTrayFunction
from .test_functions_2d.drop_wave_function import DropWaveFunction
from .test_functions_2d.easom_function import EasomFunction
from .test_functions_2d.eggholder_function import EggholderFunction
from .test_functions_2d.goldstein_price_function import GoldsteinPriceFunction
from .test_functions_2d.himmelblaus_function import HimmelblausFunction
from .test_functions_2d.hoelder_table_function import HölderTableFunction
from .test_functions_2d.langermann_function import LangermannFunction
from .test_functions_2d.levi_function_n13 import LeviFunctionN13
from .test_functions_2d.matyas_function import MatyasFunction
from .test_functions_2d.mccormick_function import McCormickFunction
from .test_functions_2d.schaffer_function_n2 import SchafferFunctionN2
from .test_functions_2d.simionescu_function import SimionescuFunction
from .test_functions_2d.three_hump_camel_function import ThreeHumpCamelFunction


from .test_functions_nd.rastrigin_function import RastriginFunction
from .test_functions_nd.rosenbrock_function import RosenbrockFunction
from .test_functions_nd.sphere_function import SphereFunction
from .test_functions_nd.styblinski_tang_function import StyblinskiTangFunction
from .test_functions_nd.griewank_function import GriewankFunction


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
    SchafferFunctionN2,
    SimionescuFunction,
    ThreeHumpCamelFunction,
]


mathematical_functions_nd = [
    GriewankFunction,
    RastriginFunction,
    RosenbrockFunction,
    SphereFunction,
    StyblinskiTangFunction,
]
