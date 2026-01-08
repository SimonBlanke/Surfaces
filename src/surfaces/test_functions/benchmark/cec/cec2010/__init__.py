# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2010 Large-Scale Global Optimization Benchmark Functions.

This module provides test functions from the CEC 2010 Special Session on
Large Scale Global Optimization. All functions have 1000 dimensions with
partial separability (groups of 50 variables).

Function Categories:
- F1-F3: Fully separable
- F4-F7: Single-group rotated
- F8-F13: Multi-group rotated (20 groups of 50)
- F14-F18: Non-separable with overlap
- F19-F20: Composition functions

References
----------
Tang, K., Li, X., Suganthan, P. N., Yang, Z., & Weise, T. (2010).
Benchmark Functions for the CEC'2010 Special Session and Competition
on Large Scale Global Optimization.
Technical Report, Nature Inspired Computation and Applications Laboratory.
"""

from ._base_cec2010 import (
    CEC2010Function,
    CEC2010SeparableFunction,
    CEC2010PartialSeparableFunction,
    CEC2010NonSeparableFunction,
    CEC2010CompositionFunction,
)
from .functions import (
    # F1-F3: Fully separable
    SeparableElliptic,
    SeparableRastrigin,
    SeparableAckley,
    # F4-F7: Single-group rotated
    SingleGroupElliptic,
    SingleGroupRastrigin,
    SingleGroupAckley,
    SingleGroupSchwefel,
    # F8-F13: Multi-group rotated
    MultiGroupElliptic,
    MultiGroupRastrigin,
    MultiGroupAckley,
    MultiGroupSchwefel,
    MultiGroupRosenbrock,
    MultiGroupGriewank,
    # F14-F18: Non-separable with overlap
    OverlapSchwefel,
    OverlapRosenbrock,
    NonSepRastrigin,
    NonSepAckley,
    NonSepGriewank,
    # F19-F20: Composition
    Composition1,
    Composition2,
    # Collections
    CEC2010_ALL,
    CEC2010_SEPARABLE,
    CEC2010_PARTIAL_SEPARABLE,
    CEC2010_NONSEPARABLE,
    CEC2010_COMPOSITION,
)

__all__ = [
    # Base classes
    "CEC2010Function",
    "CEC2010SeparableFunction",
    "CEC2010PartialSeparableFunction",
    "CEC2010NonSeparableFunction",
    "CEC2010CompositionFunction",
    # F1-F3: Fully separable
    "SeparableElliptic",
    "SeparableRastrigin",
    "SeparableAckley",
    # F4-F7: Single-group rotated
    "SingleGroupElliptic",
    "SingleGroupRastrigin",
    "SingleGroupAckley",
    "SingleGroupSchwefel",
    # F8-F13: Multi-group rotated
    "MultiGroupElliptic",
    "MultiGroupRastrigin",
    "MultiGroupAckley",
    "MultiGroupSchwefel",
    "MultiGroupRosenbrock",
    "MultiGroupGriewank",
    # F14-F18: Non-separable with overlap
    "OverlapSchwefel",
    "OverlapRosenbrock",
    "NonSepRastrigin",
    "NonSepAckley",
    "NonSepGriewank",
    # F19-F20: Composition
    "Composition1",
    "Composition2",
    # Collections
    "CEC2010_ALL",
    "CEC2010_SEPARABLE",
    "CEC2010_PARTIAL_SEPARABLE",
    "CEC2010_NONSEPARABLE",
    "CEC2010_COMPOSITION",
]
