# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2005 Benchmark Functions.

This module provides implementations of the 25 benchmark functions from the
CEC 2005 Special Session on Real-Parameter Optimization.

Functions are organized into three categories:
- Unimodal (F1-F5): Simple functions with single global optimum
- Multimodal (F6-F14): Functions with multiple local optima
- Composition (F15-F25): Hybrid functions combining multiple basic functions

References
----------
Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y.-P.,
Auger, A., & Tiwari, S. (2005). Problem definitions and evaluation
criteria for the CEC 2005 special session on real-parameter optimization.
Technical Report, Nanyang Technological University, Singapore.
"""

from .composition import (
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
    CompositionFunction9,
    CompositionFunction10,
    CompositionFunction11,
)
from .multimodal import (
    ExpandedGriewankRosenbrock,
    SchwefelProblem213,
    ShiftedRastrigin,
    ShiftedRosenbrock,
    ShiftedRotatedAckley,
    ShiftedRotatedExpandedScaffer,
    ShiftedRotatedGriewank,
    ShiftedRotatedRastrigin,
    ShiftedRotatedWeierstrass,
)
from .unimodal import (
    SchwefelProblem26,
    ShiftedRotatedElliptic,
    ShiftedSchwefel12,
    ShiftedSchwefel12Noise,
    ShiftedSphere,
)

__all__ = [
    # Unimodal (F1-F5)
    "ShiftedSphere",  # F1
    "ShiftedSchwefel12",  # F2
    "ShiftedRotatedElliptic",  # F3
    "ShiftedSchwefel12Noise",  # F4
    "SchwefelProblem26",  # F5
    # Multimodal (F6-F14)
    "ShiftedRosenbrock",  # F6
    "ShiftedRotatedGriewank",  # F7
    "ShiftedRotatedAckley",  # F8
    "ShiftedRastrigin",  # F9
    "ShiftedRotatedRastrigin",  # F10
    "ShiftedRotatedWeierstrass",  # F11
    "SchwefelProblem213",  # F12
    "ExpandedGriewankRosenbrock",  # F13
    "ShiftedRotatedExpandedScaffer",  # F14
    # Composition (F15-F25)
    "CompositionFunction1",  # F15
    "CompositionFunction2",  # F16
    "CompositionFunction3",  # F17
    "CompositionFunction4",  # F18
    "CompositionFunction5",  # F19
    "CompositionFunction6",  # F20
    "CompositionFunction7",  # F21
    "CompositionFunction8",  # F22
    "CompositionFunction9",  # F23
    "CompositionFunction10",  # F24
    "CompositionFunction11",  # F25
]

# Mapping from function ID to class
CEC2005_FUNCTIONS = {
    1: ShiftedSphere,
    2: ShiftedSchwefel12,
    3: ShiftedRotatedElliptic,
    4: ShiftedSchwefel12Noise,
    5: SchwefelProblem26,
    6: ShiftedRosenbrock,
    7: ShiftedRotatedGriewank,
    8: ShiftedRotatedAckley,
    9: ShiftedRastrigin,
    10: ShiftedRotatedRastrigin,
    11: ShiftedRotatedWeierstrass,
    12: SchwefelProblem213,
    13: ExpandedGriewankRosenbrock,
    14: ShiftedRotatedExpandedScaffer,
    15: CompositionFunction1,
    16: CompositionFunction2,
    17: CompositionFunction3,
    18: CompositionFunction4,
    19: CompositionFunction5,
    20: CompositionFunction6,
    21: CompositionFunction7,
    22: CompositionFunction8,
    23: CompositionFunction9,
    24: CompositionFunction10,
    25: CompositionFunction11,
}
