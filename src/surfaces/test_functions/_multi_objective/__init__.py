# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Multi-objective optimization test functions.

This module provides benchmark functions for multi-objective optimization,
where the goal is to find the Pareto front - the set of solutions where
no objective can be improved without worsening another.

Available Functions
-------------------
ZDT1 : Convex Pareto front, scalable dimensions
FonsecaFleming : Non-convex Pareto front, scalable dimensions
Kursawe : Disconnected Pareto front, scalable dimensions

Examples
--------
>>> from surfaces.multi_objective import ZDT1
>>> func = ZDT1(n_dim=30)
>>> result = func([0.5] + [0.0] * 29)
>>> result.shape
(2,)

>>> # Get the theoretical Pareto front
>>> front = func.pareto_front(n_points=100)
>>> front.shape
(100, 2)
"""

from ._base_multi_objective import MultiObjectiveFunction
from .fonseca_fleming import FonsecaFleming
from .kursawe import Kursawe
from .zdt1 import ZDT1

__all__ = [
    "MultiObjectiveFunction",
    "ZDT1",
    "FonsecaFleming",
    "Kursawe",
]

multi_objective_functions = [
    ZDT1,
    FonsecaFleming,
    Kursawe,
]
