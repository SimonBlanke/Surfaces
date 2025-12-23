# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .knn_ts_classifier import KNNTSClassifierFunction
from .random_forest_ts_classifier import RandomForestTSClassifierFunction

# sktime-based classifiers (require sktime)
try:
    from .ts_forest_classifier import TSForestClassifierFunction

    _HAS_SKTIME = True
except ImportError:
    _HAS_SKTIME = False

__all__ = [
    "RandomForestTSClassifierFunction",
    "KNNTSClassifierFunction",
]

if _HAS_SKTIME:
    __all__.append("TSForestClassifierFunction")
