# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2015 Learning-based Optimization Benchmark Functions.

This module provides test functions from the CEC 2015 Special Session on
Learning-based Real-Parameter Single Objective Optimization.

Function Categories:
- F1-F2: Unimodal functions
- F3-F9: Multimodal functions
- F10-F12: Hybrid functions
- F13-F15: Composition functions

References
----------
Liang, J. J., Qu, B. Y., Suganthan, P. N., & Chen, Q. (2014).
Problem definitions and evaluation criteria for the CEC 2015
competition on learning-based real-parameter single objective optimization.
Technical Report, Zhengzhou University and Nanyang Technological University.
"""

from ._base_cec2015 import CEC2015Function
from .functions import (
    # F1-F2: Unimodal
    RotatedBentCigar2015,
    RotatedDiscus2015,
    # F3-F9: Multimodal
    ShiftedRotatedWeierstrass2015,
    ShiftedRotatedSchwefel2015,
    ShiftedRotatedKatsuura2015,
    ShiftedRotatedHappyCat2015,
    ShiftedRotatedHGBat2015,
    ExpandedGriewankRosenbrock2015,
    ExpandedScafferF62015,
    # F10-F12: Hybrid
    HybridFunction1_2015,
    HybridFunction2_2015,
    HybridFunction3_2015,
    # F13-F15: Composition
    CompositionFunction1_2015,
    CompositionFunction2_2015,
    CompositionFunction3_2015,
    # Collections
    CEC2015_ALL,
    CEC2015_UNIMODAL,
    CEC2015_MULTIMODAL,
    CEC2015_HYBRID,
    CEC2015_COMPOSITION,
)

__all__ = [
    # Base class
    "CEC2015Function",
    # F1-F2: Unimodal
    "RotatedBentCigar2015",
    "RotatedDiscus2015",
    # F3-F9: Multimodal
    "ShiftedRotatedWeierstrass2015",
    "ShiftedRotatedSchwefel2015",
    "ShiftedRotatedKatsuura2015",
    "ShiftedRotatedHappyCat2015",
    "ShiftedRotatedHGBat2015",
    "ExpandedGriewankRosenbrock2015",
    "ExpandedScafferF62015",
    # F10-F12: Hybrid
    "HybridFunction1_2015",
    "HybridFunction2_2015",
    "HybridFunction3_2015",
    # F13-F15: Composition
    "CompositionFunction1_2015",
    "CompositionFunction2_2015",
    "CompositionFunction3_2015",
    # Collections
    "CEC2015_ALL",
    "CEC2015_UNIMODAL",
    "CEC2015_MULTIMODAL",
    "CEC2015_HYBRID",
    "CEC2015_COMPOSITION",
]
