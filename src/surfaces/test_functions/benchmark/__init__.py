# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Established academic benchmark suites.

This module contains well-known benchmark suites from the optimization community:
- bbob: COCO/BBOB (Black-Box Optimization Benchmarking) - 24 functions
- cec: CEC Competition benchmarks (2013, 2014, 2017)
"""

# BBOB functions
from .bbob import (
    BBOB_FUNCTIONS,
    # Low/Moderate Conditioning (f6-f9)
    AttractiveSector,
    BBOBFunction,
    BentCigar,
    BuecheRastrigin,
    DifferentPowers,
    Discus,
    # High Conditioning & Unimodal (f10-f14)
    EllipsoidalRotated,
    EllipsoidalSeparable,
    Gallagher21,
    Gallagher101,
    GriewankRosenbrock,
    Katsuura,
    LinearSlope,
    LunacekBiRastrigin,
    # Multimodal with Adequate Global Structure (f15-f19)
    RastriginRotated,
    RastriginSeparable,
    RosenbrockOriginal,
    RosenbrockRotated,
    SchaffersF7,
    SchaffersF7Ill,
    # Multimodal with Weak Global Structure (f20-f24)
    Schwefel,
    SharpRidge,
    # Separable (f1-f5)
    Sphere,
    StepEllipsoidal,
    Weierstrass,
    bbob_functions,
)

# CEC functions (require cec data package)
try:
    from .cec import CECFunction

    cec_functions = []  # CEC functions are loaded dynamically
except ImportError:
    cec_functions = []

__all__ = [
    # Base classes
    "BBOBFunction",
    # BBOB - Separable
    "Sphere",
    "EllipsoidalSeparable",
    "RastriginSeparable",
    "BuecheRastrigin",
    "LinearSlope",
    # BBOB - Low/Moderate Conditioning
    "AttractiveSector",
    "StepEllipsoidal",
    "RosenbrockOriginal",
    "RosenbrockRotated",
    # BBOB - High Conditioning & Unimodal
    "EllipsoidalRotated",
    "Discus",
    "BentCigar",
    "SharpRidge",
    "DifferentPowers",
    # BBOB - Multimodal Adequate
    "RastriginRotated",
    "Weierstrass",
    "SchaffersF7",
    "SchaffersF7Ill",
    "GriewankRosenbrock",
    # BBOB - Multimodal Weak
    "Schwefel",
    "Gallagher101",
    "Gallagher21",
    "Katsuura",
    "LunacekBiRastrigin",
    # Function lists
    "bbob_functions",
    "BBOB_FUNCTIONS",
    "cec_functions",
]

benchmark_functions = bbob_functions + cec_functions
