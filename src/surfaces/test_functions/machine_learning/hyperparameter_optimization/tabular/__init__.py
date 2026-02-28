# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .classification import (
    DecisionTreeClassifierFunction,
    GradientBoostingClassifierFunction,
    KNeighborsClassifierFunction,
    LightGBMClassifierFunction,
    RandomForestClassifierFunction,
    SVMClassifierFunction,
    XGBoostClassifierFunction,
)
from .regression import (
    DecisionTreeRegressorFunction,
    GradientBoostingRegressorFunction,
    KNeighborsRegressorFunction,
    LightGBMRegressorFunction,
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
    "XGBoostClassifierFunction",
    "LightGBMClassifierFunction",
    # Regression
    "DecisionTreeRegressorFunction",
    "GradientBoostingRegressorFunction",
    "KNeighborsRegressorFunction",
    "RandomForestRegressorFunction",
    "SVMRegressorFunction",
    "LightGBMRegressorFunction",
]
