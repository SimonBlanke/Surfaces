# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Constrained algebraic test functions (engineering design problems)."""

from ._base_engineering_function import EngineeringFunction
from .cantilever_beam import CantileverBeamFunction
from .pressure_vessel import PressureVesselFunction
from .tension_compression_spring import TensionCompressionSpringFunction
from .three_bar_truss import ThreeBarTrussFunction
from .welded_beam import WeldedBeamFunction

__all__ = [
    "EngineeringFunction",
    "CantileverBeamFunction",
    "PressureVesselFunction",
    "TensionCompressionSpringFunction",
    "ThreeBarTrussFunction",
    "WeldedBeamFunction",
]

constrained_functions = [
    CantileverBeamFunction,
    PressureVesselFunction,
    TensionCompressionSpringFunction,
    ThreeBarTrussFunction,
    WeldedBeamFunction,
]
