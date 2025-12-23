# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .random_forest_image_classifier import RandomForestImageClassifierFunction
from .svm_image_classifier import SVMImageClassifierFunction

# CNN classifiers (require tensorflow)
try:
    from .deep_cnn_classifier import DeepCNNClassifierFunction
    from .simple_cnn_classifier import SimpleCNNClassifierFunction

    _HAS_TENSORFLOW = True
except ImportError:
    _HAS_TENSORFLOW = False

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
