# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .tabular_classifiers import KNeighborsClassifierFunction
from .tabular_regressors import (
    GradientBoostingRegressorFunction,
    KNeighborsRegressorFunction,
)

__all__ = [
    "KNeighborsClassifierFunction",
    "GradientBoostingRegressorFunction",
    "KNeighborsRegressorFunction",
]

machine_learning_functions = [
    KNeighborsClassifierFunction,
    GradientBoostingRegressorFunction,
    KNeighborsRegressorFunction,
]
