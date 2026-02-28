# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .classification import (
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
