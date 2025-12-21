# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2017 Benchmark Functions.

This module provides the 30 benchmark functions from the CEC 2017 competition
on single objective bound constrained real-parameter numerical optimization.

Functions are organized into categories:
- Simple (F1-F10): Shifted and rotated classical functions
- Hybrid (F11-F20): Combinations of basic functions with variable partitioning
- Composition (F21-F30): Complex landscapes from multiple functions

Note: F2 has been deprecated from the official CEC 2017 benchmark suite.

Reference:
    Awad, N. H., Ali, M. Z., Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2016).
    Problem definitions and evaluation criteria for the CEC 2017 special
    session and competition on single objective bound constrained real-parameter
    numerical optimization.
"""

from ._base_cec2017 import CEC2017Function
from .simple import (
    ShiftedRotatedBentCigar,
    ShiftedRotatedLevy,
    ShiftedRotatedLunacekBiRastrigin,
    ShiftedRotatedNonContRastrigin,
    ShiftedRotatedRastrigin,
    ShiftedRotatedRosenbrock,
    ShiftedRotatedSchafferF7,
    ShiftedRotatedSchwefel,
    ShiftedRotatedSumDiffPow,
    ShiftedRotatedZakharov,
)

__all__ = [
    "CEC2017Function",
    # Simple (F1-F10)
    "ShiftedRotatedBentCigar",
    "ShiftedRotatedSumDiffPow",
    "ShiftedRotatedZakharov",
    "ShiftedRotatedRosenbrock",
    "ShiftedRotatedRastrigin",
    "ShiftedRotatedSchafferF7",
    "ShiftedRotatedLunacekBiRastrigin",
    "ShiftedRotatedNonContRastrigin",
    "ShiftedRotatedLevy",
    "ShiftedRotatedSchwefel",
]
