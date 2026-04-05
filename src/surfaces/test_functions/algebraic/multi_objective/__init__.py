# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Multi-objective optimization test functions.

This module provides benchmark functions for multi-objective optimization,
where the goal is to find the Pareto front -- the set of solutions where
no objective can be improved without worsening another.

Available Families
------------------
ZDT : Bi-objective problems with various Pareto front geometries (ZDT1-4, ZDT6)
DTLZ : Scalable problems with configurable number of objectives (DTLZ1-7)
WFG : Transformation-based problems with configurable pipelines (WFG1-9)
FonsecaFleming : Non-convex Pareto front, scalable dimensions
Kursawe : Disconnected Pareto front, scalable dimensions

Examples
--------
>>> from surfaces.test_functions.algebraic.multi_objective import ZDT1
>>> func = ZDT1(n_dim=30)
>>> result = func([0.5] + [0.0] * 29)
>>> result.shape
(2,)

>>> from surfaces.test_functions.algebraic.multi_objective import DTLZ2
>>> func = DTLZ2(n_objectives=3)
>>> front = func.pareto_front(n_points=100)
>>> front.shape
(100, 3)
"""

from ._base_multi_objective import BaseMultiObjectiveTestFunction, MultiObjectiveFunction
from .dtlz import (
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ6,
    DTLZ7,
    dtlz_functions,
)
from .fonseca_fleming import FonsecaFleming
from .kursawe import Kursawe
from .wfg import (
    WFG1,
    WFG2,
    WFG3,
    WFG4,
    WFG5,
    WFG6,
    WFG7,
    WFG8,
    WFG9,
    wfg_functions,
)
from .zdt import (
    ZDT1,
    ZDT2,
    ZDT3,
    ZDT4,
    ZDT6,
    zdt_functions,
)

__all__ = [
    "BaseMultiObjectiveTestFunction",
    "MultiObjectiveFunction",
    # ZDT
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "ZDT4",
    "ZDT6",
    # DTLZ
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
    # WFG
    "WFG1",
    "WFG2",
    "WFG3",
    "WFG4",
    "WFG5",
    "WFG6",
    "WFG7",
    "WFG8",
    "WFG9",
    # Others
    "FonsecaFleming",
    "Kursawe",
]

multi_objective_functions = (
    zdt_functions + dtlz_functions + wfg_functions + [FonsecaFleming, Kursawe]
)
