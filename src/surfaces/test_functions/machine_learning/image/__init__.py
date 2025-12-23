# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .classification import (
    RandomForestImageClassifierFunction,
    SVMImageClassifierFunction,
)

# CNN classifiers (require tensorflow)
try:
    from .classification import DeepCNNClassifierFunction, SimpleCNNClassifierFunction

    _HAS_TENSORFLOW = True
except ImportError:
    _HAS_TENSORFLOW = False

# XGBoost classifier (requires xgboost)
try:
    from .classification import XGBoostImageClassifierFunction

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

__all__ = [
    "SVMImageClassifierFunction",
    "RandomForestImageClassifierFunction",
]

if _HAS_TENSORFLOW:
    __all__.extend(
        [
            "SimpleCNNClassifierFunction",
            "DeepCNNClassifierFunction",
        ]
    )

if _HAS_XGBOOST:
    __all__.append("XGBoostImageClassifierFunction")
