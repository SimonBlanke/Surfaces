# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .decision_tree_regressor import DecisionTreeRegressorFunction
from .gradient_boosting_regressor import GradientBoostingRegressorFunction
from .k_neighbors_regressor import KNeighborsRegressorFunction
from .random_forest_regressor import RandomForestRegressorFunction
from .svm_regressor import SVMRegressorFunction

__all__ = [
    "DecisionTreeRegressorFunction",
    "GradientBoostingRegressorFunction",
    "KNeighborsRegressorFunction",
    "RandomForestRegressorFunction",
    "SVMRegressorFunction",
]
