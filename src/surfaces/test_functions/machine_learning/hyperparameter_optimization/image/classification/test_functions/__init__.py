# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .deep_cnn_classifier import DeepCNNClassifierFunction
from .random_forest_image_classifier import RandomForestImageClassifierFunction
from .simple_cnn_classifier import SimpleCNNClassifierFunction
from .svm_image_classifier import SVMImageClassifierFunction
from .xgboost_image_classifier import XGBoostImageClassifierFunction

__all__ = [
    "SVMImageClassifierFunction",
    "RandomForestImageClassifierFunction",
    "SimpleCNNClassifierFunction",
    "DeepCNNClassifierFunction",
    "XGBoostImageClassifierFunction",
]
