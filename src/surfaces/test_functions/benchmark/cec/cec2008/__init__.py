# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2008 Large-Scale Global Optimization Benchmark Functions.

This module provides the 7 functions from the CEC 2008 Special Session on
Large Scale Global Optimization. All functions have 1000 dimensions.

Function overview:
- F1: Shifted Sphere (separable, unimodal)
- F2: Shifted Schwefel 2.21 (separable, unimodal)
- F3: Shifted Rosenbrock (non-separable, multimodal)
- F4: Shifted Rastrigin (separable, multimodal)
- F5: Shifted Griewank (non-separable, multimodal)
- F6: Shifted Ackley (separable, multimodal)
- F7: Fast Fractal Double Dip (non-separable, multimodal)

References
----------
Tang, K., Li, X., Suganthan, P. N., Yang, Z., & Weise, T. (2008).
Benchmark Functions for the CEC'2008 Special Session and Competition
on Large Scale Global Optimization.
Technical Report, Nature Inspired Computation and Applications Laboratory.

Examples
--------
>>> from surfaces.test_functions.benchmark.cec.cec2008 import ShiftedSphere2008
>>> func = ShiftedSphere2008()
>>> func.n_dim
1000
>>> func.f_global
0.0
"""

from ._base_cec2008 import (
    CEC2008Function,
    CEC2008SeparableFunction,
    CEC2008NonSeparableFunction,
)
from .functions import (
    ShiftedSphere2008,
    ShiftedSchwefel221,
    ShiftedRosenbrock2008,
    ShiftedRastrigin2008,
    ShiftedGriewank2008,
    ShiftedAckley2008,
    FastFractalDoubleDip,
    CEC2008_ALL,
    CEC2008_SEPARABLE,
    CEC2008_NONSEPARABLE,
)

__all__ = [
    # Base classes
    "CEC2008Function",
    "CEC2008SeparableFunction",
    "CEC2008NonSeparableFunction",
    # Functions
    "ShiftedSphere2008",
    "ShiftedSchwefel221",
    "ShiftedRosenbrock2008",
    "ShiftedRastrigin2008",
    "ShiftedGriewank2008",
    "ShiftedAckley2008",
    "FastFractalDoubleDip",
    # Lists
    "CEC2008_ALL",
    "CEC2008_SEPARABLE",
    "CEC2008_NONSEPARABLE",
]
