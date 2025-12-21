# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2014 Benchmark Functions.

This module provides the 30 benchmark functions from the CEC 2014 competition
on single objective real-parameter numerical optimization.

Functions are organized into categories:
- Unimodal (F1-F3): Single global optimum
- Simple Multimodal (F4-F16): Multiple local optima
- Hybrid (F17-F22): Combinations of basic functions
- Composition (F23-F30): Complex landscapes from multiple functions

Reference:
    Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013).
    Problem definitions and evaluation criteria for the CEC 2014 special
    session and competition on single objective real-parameter numerical
    optimization. Computational Intelligence Laboratory, Zhengzhou University.
"""

from ._base_cec2014 import CEC2014Function
from .composition import (
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
)
from .hybrid import (
    HybridFunction1,
    HybridFunction2,
    HybridFunction3,
    HybridFunction4,
    HybridFunction5,
    HybridFunction6,
)
from .multimodal import (
    ShiftedRastrigin,
    ShiftedRotatedAckley,
    ShiftedRotatedExpandedGriewankRosenbrock,
    ShiftedRotatedExpandedScafferF6,
    ShiftedRotatedGriewank,
    ShiftedRotatedHappyCat,
    ShiftedRotatedHGBat,
    ShiftedRotatedKatsuura,
    ShiftedRotatedRastrigin,
    ShiftedRotatedRosenbrock,
    ShiftedRotatedSchwefel,
    ShiftedRotatedWeierstrass,
    ShiftedSchwefel,
)
from .unimodal import (
    RotatedBentCigar,
    RotatedDiscus,
    RotatedHighConditionedElliptic,
)

__all__ = [
    "CEC2014Function",
    # Unimodal (F1-F3)
    "RotatedHighConditionedElliptic",
    "RotatedBentCigar",
    "RotatedDiscus",
    # Multimodal (F4-F16)
    "ShiftedRotatedRosenbrock",
    "ShiftedRotatedAckley",
    "ShiftedRotatedWeierstrass",
    "ShiftedRotatedGriewank",
    "ShiftedRastrigin",
    "ShiftedRotatedRastrigin",
    "ShiftedSchwefel",
    "ShiftedRotatedSchwefel",
    "ShiftedRotatedKatsuura",
    "ShiftedRotatedHappyCat",
    "ShiftedRotatedHGBat",
    "ShiftedRotatedExpandedGriewankRosenbrock",
    "ShiftedRotatedExpandedScafferF6",
    # Hybrid (F17-F22)
    "HybridFunction1",
    "HybridFunction2",
    "HybridFunction3",
    "HybridFunction4",
    "HybridFunction5",
    "HybridFunction6",
    # Composition (F23-F30)
    "CompositionFunction1",
    "CompositionFunction2",
    "CompositionFunction3",
    "CompositionFunction4",
    "CompositionFunction5",
    "CompositionFunction6",
    "CompositionFunction7",
    "CompositionFunction8",
]
