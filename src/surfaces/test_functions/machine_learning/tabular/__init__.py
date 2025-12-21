# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .classification import KNeighborsClassifierFunction
from .regression import GradientBoostingRegressorFunction, KNeighborsRegressorFunction

__all__ = [
    "KNeighborsClassifierFunction",
    "KNeighborsRegressorFunction",
    "GradientBoostingRegressorFunction",
]
