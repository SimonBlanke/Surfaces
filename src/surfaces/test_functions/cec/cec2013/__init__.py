# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2013 Benchmark Functions.

This module provides the 28 benchmark functions from the CEC 2013 competition
on single objective real-parameter numerical optimization.

Functions are organized into categories:
- Unimodal (F1-F5): Single global optimum
- Multimodal (F6-F20): Multiple local optima
- Composition (F21-F28): Complex landscapes from multiple functions

Reference:
    Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernandez-Diaz, A. G. (2013).
    Problem definitions and evaluation criteria for the CEC 2013 special
    session on real-parameter optimization. Computational Intelligence
    Laboratory, Zhengzhou University.
"""

from ._base_cec2013 import CEC2013Function
from .functions import (
    # Composition (F21-F28)
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
    DifferentPowers,
    LunacekBiRastrigin,
    Rastrigin,
    RotatedAckley,
    RotatedBentCigar,
    RotatedDiscus,
    RotatedExpandedGriewankRosenbrock,
    RotatedExpandedScafferF6,
    RotatedGriewank,
    RotatedHighConditionedElliptic,
    RotatedKatsuura,
    RotatedLunacekBiRastrigin,
    RotatedRastrigin,
    # Multimodal (F6-F20)
    RotatedRosenbrock,
    RotatedSchafferF7,
    RotatedSchwefel,
    RotatedWeierstrass,
    Schwefel,
    # Unimodal (F1-F5)
    Sphere,
    StepRastrigin,
)

__all__ = [
    "CEC2013Function",
    # Unimodal (F1-F5)
    "Sphere",
    "RotatedHighConditionedElliptic",
    "RotatedBentCigar",
    "RotatedDiscus",
    "DifferentPowers",
    # Multimodal (F6-F20)
    "RotatedRosenbrock",
    "RotatedSchafferF7",
    "RotatedAckley",
    "RotatedWeierstrass",
    "RotatedGriewank",
    "Rastrigin",
    "RotatedRastrigin",
    "StepRastrigin",
    "Schwefel",
    "RotatedSchwefel",
    "RotatedKatsuura",
    "LunacekBiRastrigin",
    "RotatedLunacekBiRastrigin",
    "RotatedExpandedGriewankRosenbrock",
    "RotatedExpandedScafferF6",
    # Composition (F21-F28)
    "CompositionFunction1",
    "CompositionFunction2",
    "CompositionFunction3",
    "CompositionFunction4",
    "CompositionFunction5",
    "CompositionFunction6",
    "CompositionFunction7",
    "CompositionFunction8",
]
