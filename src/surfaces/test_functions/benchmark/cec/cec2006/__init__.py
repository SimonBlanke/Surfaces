# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2006 Constrained Benchmark Functions.

This module provides the 24 constrained optimization test functions from the
CEC 2006 Special Session on Constrained Real-Parameter Optimization.

Unlike CEC 2013/2014/2017, these functions:
- Have fixed dimensions (not scalable)
- Include inequality and/or equality constraints
- Use original function forms (no shift/rotation)
- Have problem-specific bounds

References
----------
Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M.,
Suganthan, P. N., Coello, C. A. C., & Deb, K. (2006).
Problem definitions and evaluation criteria for the CEC 2006
special session on constrained real-parameter optimization.
Technical Report, Nanyang Technological University, Singapore.

Examples
--------
>>> from surfaces.test_functions.benchmark.cec.cec2006 import G01
>>> func = G01()
>>> func.n_dim
13
>>> func.n_constraints
9
>>> result = func(func.x_global)
>>> func.is_feasible(func.x_global)
True
"""

from ._base_cec2006 import CEC2006Function
from .functions import (
    G01,
    G02,
    G03,
    G04,
    G05,
    G06,
    G07,
    G08,
    G09,
    G10,
    G11,
    G12,
    G13,
    G14,
    G15,
    G16,
    G17,
    G18,
    G19,
    G20,
    G21,
    G22,
    G23,
    G24,
    CEC2006_ALL,
)

__all__ = [
    # Base class
    "CEC2006Function",
    # Functions
    "G01",
    "G02",
    "G03",
    "G04",
    "G05",
    "G06",
    "G07",
    "G08",
    "G09",
    "G10",
    "G11",
    "G12",
    "G13",
    "G14",
    "G15",
    "G16",
    "G17",
    "G18",
    "G19",
    "G20",
    "G21",
    "G22",
    "G23",
    "G24",
    # List
    "CEC2006_ALL",
]
