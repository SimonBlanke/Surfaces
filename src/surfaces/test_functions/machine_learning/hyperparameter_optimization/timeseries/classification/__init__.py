# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .test_functions import (
    KNNTSClassifierFunction,
    RandomForestTSClassifierFunction,
)

# sktime-based classifiers (require sktime)
try:
    from .test_functions import TSForestClassifierFunction

    _HAS_SKTIME = True
except ImportError:
    _HAS_SKTIME = False

__all__ = [
    "RandomForestTSClassifierFunction",
    "KNNTSClassifierFunction",
]

if _HAS_SKTIME:
    __all__.append("TSForestClassifierFunction")
