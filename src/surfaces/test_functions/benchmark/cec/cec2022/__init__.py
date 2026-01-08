# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2022 Single Objective Bound Constrained Benchmark Functions.

This module provides all 12 functions from the CEC 2022 Special Session
and Competition on Single Objective Bound Constrained Numerical Optimization.

Functions
---------
F1 : Shifted and Full Rotated Zakharov (Unimodal)
F2 : Shifted and Rotated Rosenbrock's (Multimodal)
F3 : Shifted and Full Rotated Expanded Schaffer's F7 (Multimodal)
F4 : Shifted and Rotated Non-Continuous Rastrigin's (Multimodal)
F5 : Shifted and Rotated Levy (Multimodal)
F6 : Hybrid Function 1 (Bent Cigar + HGBat + Rastrigin + Schwefel)
F7 : Hybrid Function 2 (HGBat + Katsuura + Ackley + Rastrigin + Schwefel + Schaffer F7)
F8 : Hybrid Function 3 (Katsuura + HappyCat + Griewank-Rosenbrock + Schwefel + Ackley)
F9 : Composition Function 1 (Rosenbrock + Elliptic + Bent Cigar + Discus)
F10 : Composition Function 2 (Schwefel + Rastrigin + HGBat)
F11 : Composition Function 3 (Expanded Schaffer F6 + Schwefel + Griewank + Rosenbrock + Rastrigin)
F12 : Composition Function 4 (HGBat + Rastrigin + Schwefel + Bent Cigar + Elliptic + Schaffer F6)

All functions have:
- Search bounds: [-100, 100]^D
- Supported dimensions: 10, 20

References
----------
Abhishek Kumar, Kenneth V. Price, Ali Wagdy Mohamed, Anas A. Hadi,
P. N. Suganthan (2021). Problem definitions and evaluation criteria for
the CEC 2022 special session and competition on single objective bound
constrained numerical optimization.
"""

from ._base_cec2022 import CEC2022Function
from .composition import (
    CompositionFunction1_2022,
    CompositionFunction2_2022,
    CompositionFunction3_2022,
    CompositionFunction4_2022,
    CEC2022_COMPOSITION,
)
from .functions import (
    ShiftedRotatedZakharov2022,
    ShiftedRotatedRosenbrock2022,
    ShiftedRotatedExpandedSchafferF72022,
    ShiftedRotatedNonContRastrigin2022,
    ShiftedRotatedLevy2022,
    CEC2022_BASIC,
)
from .hybrid import (
    HybridFunction1_2022,
    HybridFunction2_2022,
    HybridFunction3_2022,
    CEC2022_HYBRID,
)

# Complete collection of all CEC 2022 functions
CEC2022_ALL = [
    ShiftedRotatedZakharov2022,  # F1
    ShiftedRotatedRosenbrock2022,  # F2
    ShiftedRotatedExpandedSchafferF72022,  # F3
    ShiftedRotatedNonContRastrigin2022,  # F4
    ShiftedRotatedLevy2022,  # F5
    HybridFunction1_2022,  # F6
    HybridFunction2_2022,  # F7
    HybridFunction3_2022,  # F8
    CompositionFunction1_2022,  # F9
    CompositionFunction2_2022,  # F10
    CompositionFunction3_2022,  # F11
    CompositionFunction4_2022,  # F12
]

__all__ = [
    # Base class
    "CEC2022Function",
    # Basic functions (F1-F5)
    "ShiftedRotatedZakharov2022",
    "ShiftedRotatedRosenbrock2022",
    "ShiftedRotatedExpandedSchafferF72022",
    "ShiftedRotatedNonContRastrigin2022",
    "ShiftedRotatedLevy2022",
    # Hybrid functions (F6-F8)
    "HybridFunction1_2022",
    "HybridFunction2_2022",
    "HybridFunction3_2022",
    # Composition functions (F9-F12)
    "CompositionFunction1_2022",
    "CompositionFunction2_2022",
    "CompositionFunction3_2022",
    "CompositionFunction4_2022",
    # Collections
    "CEC2022_ALL",
    "CEC2022_BASIC",
    "CEC2022_HYBRID",
    "CEC2022_COMPOSITION",
]
