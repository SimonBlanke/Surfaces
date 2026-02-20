# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .test_functions import (
    DeepCNNClassifierFunction,
    RandomForestImageClassifierFunction,
    SimpleCNNClassifierFunction,
    SVMImageClassifierFunction,
    XGBoostImageClassifierFunction,
)

__all__ = [
    "SVMImageClassifierFunction",
    "RandomForestImageClassifierFunction",
    "SimpleCNNClassifierFunction",
    "DeepCNNClassifierFunction",
    "XGBoostImageClassifierFunction",
]
