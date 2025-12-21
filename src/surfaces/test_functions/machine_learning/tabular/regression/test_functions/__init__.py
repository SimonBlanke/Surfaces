# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .gradient_boosting_regressor import GradientBoostingRegressorFunction
from .k_neighbors_regressor import KNeighborsRegressorFunction

__all__ = [
    "KNeighborsRegressorFunction",
    "GradientBoostingRegressorFunction",
]
