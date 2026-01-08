# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2021 Single Objective Bound Constrained Benchmark Functions.

This module provides all 10 functions from the CEC 2021 Special Session
and Competition on Single Objective Bound Constrained Numerical Optimization.

Functions are identical in structure to CEC 2020 but use different
shift and rotation data.

Functions
---------
F1 : Shifted and Rotated Bent Cigar (Unimodal)
F2 : Shifted and Rotated Schwefel's (Multimodal)
F3 : Shifted and Rotated Lunacek Bi-Rastrigin (Multimodal)
F4 : Expanded Griewank plus Rosenbrock (Multimodal)
F5 : Hybrid Function 1 (Schwefel + Rastrigin + Elliptic)
F6 : Hybrid Function 2 (Schaffer F6 + HGBat + Rosenbrock + Schwefel)
F7 : Hybrid Function 3 (Schaffer F6 + HGBat + Rosenbrock + Schwefel + Elliptic)
F8 : Composition Function 1 (Rastrigin + Griewank + Schwefel)
F9 : Composition Function 2 (Ackley + Elliptic + Griewank + Rastrigin)
F10 : Composition Function 3 (Rastrigin + HappyCat + Ackley + Discus + Rosenbrock)

All functions have:
- Search bounds: [-100, 100]^D
- Supported dimensions: 10, 20

References
----------
Mohamed, A. W., Hadi, A. A., Mohamed, A. K., Agrawal, P., Kumar, A., &
Suganthan, P. N. (2020). Problem definitions and evaluation criteria for
the CEC 2021 special session and competition on single objective bound
constrained numerical optimization.
"""

from ._base_cec2021 import CEC2021Function
from .composition import (
    CompositionFunction1_2021,
    CompositionFunction2_2021,
    CompositionFunction3_2021,
    CEC2021_COMPOSITION,
)
from .functions import (
    ShiftedRotatedBentCigar2021,
    ShiftedRotatedSchwefel2021,
    ShiftedRotatedLunacekBiRastrigin2021,
    ExpandedGriewankRosenbrock2021,
    CEC2021_BASIC,
)
from .hybrid import (
    HybridFunction1_2021,
    HybridFunction2_2021,
    HybridFunction3_2021,
    CEC2021_HYBRID,
)

# Complete collection of all CEC 2021 functions
CEC2021_ALL = [
    ShiftedRotatedBentCigar2021,  # F1
    ShiftedRotatedSchwefel2021,  # F2
    ShiftedRotatedLunacekBiRastrigin2021,  # F3
    ExpandedGriewankRosenbrock2021,  # F4
    HybridFunction1_2021,  # F5
    HybridFunction2_2021,  # F6
    HybridFunction3_2021,  # F7
    CompositionFunction1_2021,  # F8
    CompositionFunction2_2021,  # F9
    CompositionFunction3_2021,  # F10
]

__all__ = [
    # Base class
    "CEC2021Function",
    # Basic functions (F1-F4)
    "ShiftedRotatedBentCigar2021",
    "ShiftedRotatedSchwefel2021",
    "ShiftedRotatedLunacekBiRastrigin2021",
    "ExpandedGriewankRosenbrock2021",
    # Hybrid functions (F5-F7)
    "HybridFunction1_2021",
    "HybridFunction2_2021",
    "HybridFunction3_2021",
    # Composition functions (F8-F10)
    "CompositionFunction1_2021",
    "CompositionFunction2_2021",
    "CompositionFunction3_2021",
    # Collections
    "CEC2021_ALL",
    "CEC2021_BASIC",
    "CEC2021_HYBRID",
    "CEC2021_COMPOSITION",
]
