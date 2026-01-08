# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2019 100-Digit Challenge benchmark functions.

This module provides all 10 functions from the CEC 2019 Special Session
on 100-Digit Challenge on Single Objective Numerical Optimization.

Functions
---------
F1 (StornsChebyshev) : D=9, bounds=[-8192, 8192]
F2 (InverseHilbert) : D=16, bounds=[-16384, 16384]
F3 (LennardJones) : D=18, bounds=[-4, 4]
F4-F10 (Shifted/Rotated) : D=10, bounds=[-100, 100]

All functions have global optimum f* = 1.0.

References
----------
Price, K. V., Awad, N. H., Ali, M. Z., & Suganthan, P. N. (2018).
Problem definitions and evaluation criteria for the 100-Digit Challenge
special session and competition on single objective numerical optimization.
Technical Report, Nanyang Technological University.
"""

from ._base_cec2019 import CEC2019Function
from .functions import (
    # Special functions (different dimensions)
    StornsChebyshev,
    InverseHilbert,
    LennardJones,
    # Standard shifted/rotated functions (D=10)
    ShiftedRotatedRastrigin2019,
    ShiftedRotatedGriewank2019,
    ShiftedRotatedWeierstrass2019,
    ShiftedRotatedSchwefel2019,
    ExpandedScafferF62019,
    ShiftedRotatedHappyCat2019,
    ShiftedRotatedAckley2019,
    # Collections
    CEC2019_ALL,
    CEC2019_SPECIAL,
    CEC2019_STANDARD,
)

__all__ = [
    # Base class
    "CEC2019Function",
    # Special functions
    "StornsChebyshev",
    "InverseHilbert",
    "LennardJones",
    # Standard functions
    "ShiftedRotatedRastrigin2019",
    "ShiftedRotatedGriewank2019",
    "ShiftedRotatedWeierstrass2019",
    "ShiftedRotatedSchwefel2019",
    "ExpandedScafferF62019",
    "ShiftedRotatedHappyCat2019",
    "ShiftedRotatedAckley2019",
    # Collections
    "CEC2019_ALL",
    "CEC2019_SPECIAL",
    "CEC2019_STANDARD",
]
