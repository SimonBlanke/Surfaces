# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .tabular import (
    KNeighborsClassifierFunction,
    KNeighborsRegressorFunction,
    GradientBoostingRegressorFunction,
)


__all__ = [
    "KNeighborsClassifierFunction",
    "KNeighborsRegressorFunction",
    "GradientBoostingRegressorFunction",
]


machine_learning_functions = [
    KNeighborsClassifierFunction,
    GradientBoostingRegressorFunction,
    KNeighborsRegressorFunction,
]
