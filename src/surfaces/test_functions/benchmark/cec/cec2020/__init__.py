# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2020 Single Objective Bound Constrained Benchmark Functions.

This module provides all 10 functions from the CEC 2020 Special Session
and Competition on Single Objective Bound Constrained Numerical Optimization.

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
- Supported dimensions: 5, 10, 15, 20

References
----------
Yue, C. T., Price, K. V., Suganthan, P. N., Liang, J. J., Ali, M. Z.,
Qu, B. Y., Awad, N. H., & Biswas, P. P. (2019). Problem definitions and
evaluation criteria for the CEC 2020 special session and competition on
single objective bound constrained numerical optimization.
"""

from ._base_cec2020 import CEC2020Function
from .composition import (
    CompositionFunction1_2020,
    CompositionFunction2_2020,
    CompositionFunction3_2020,
    CEC2020_COMPOSITION,
)
from .functions import (
    ShiftedRotatedBentCigar2020,
    ShiftedRotatedSchwefel2020,
    ShiftedRotatedLunacekBiRastrigin2020,
    ExpandedGriewankRosenbrock2020,
    CEC2020_BASIC,
)
from .hybrid import (
    HybridFunction1_2020,
    HybridFunction2_2020,
    HybridFunction3_2020,
    CEC2020_HYBRID,
)

# Complete collection of all CEC 2020 functions
CEC2020_ALL = [
    ShiftedRotatedBentCigar2020,  # F1
    ShiftedRotatedSchwefel2020,  # F2
    ShiftedRotatedLunacekBiRastrigin2020,  # F3
    ExpandedGriewankRosenbrock2020,  # F4
    HybridFunction1_2020,  # F5
    HybridFunction2_2020,  # F6
    HybridFunction3_2020,  # F7
    CompositionFunction1_2020,  # F8
    CompositionFunction2_2020,  # F9
    CompositionFunction3_2020,  # F10
]

__all__ = [
    # Base class
    "CEC2020Function",
    # Basic functions (F1-F4)
    "ShiftedRotatedBentCigar2020",
    "ShiftedRotatedSchwefel2020",
    "ShiftedRotatedLunacekBiRastrigin2020",
    "ExpandedGriewankRosenbrock2020",
    # Hybrid functions (F5-F7)
    "HybridFunction1_2020",
    "HybridFunction2_2020",
    "HybridFunction3_2020",
    # Composition functions (F8-F10)
    "CompositionFunction1_2020",
    "CompositionFunction2_2020",
    "CompositionFunction3_2020",
    # Collections
    "CEC2020_ALL",
    "CEC2020_BASIC",
    "CEC2020_HYBRID",
    "CEC2020_COMPOSITION",
]
