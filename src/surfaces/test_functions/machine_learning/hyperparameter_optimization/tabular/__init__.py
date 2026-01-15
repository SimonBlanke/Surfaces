# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .classification import (
    DecisionTreeClassifierFunction,
    GradientBoostingClassifierFunction,
    KNeighborsClassifierFunction,
    RandomForestClassifierFunction,
    SVMClassifierFunction,
)
from .regression import (
    DecisionTreeRegressorFunction,
    GradientBoostingRegressorFunction,
    KNeighborsRegressorFunction,
    RandomForestRegressorFunction,
    SVMRegressorFunction,
)

__all__ = [
    # Classification
    "DecisionTreeClassifierFunction",
    "GradientBoostingClassifierFunction",
    "KNeighborsClassifierFunction",
    "RandomForestClassifierFunction",
    "SVMClassifierFunction",
    # Regression
    "DecisionTreeRegressorFunction",
    "GradientBoostingRegressorFunction",
    "KNeighborsRegressorFunction",
    "RandomForestRegressorFunction",
    "SVMRegressorFunction",
]
