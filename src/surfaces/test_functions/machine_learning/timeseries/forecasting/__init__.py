# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .test_functions import (
    GradientBoostingForecasterFunction,
    RandomForestForecasterFunction,
)

# sktime-based forecasters (require sktime)
try:
    from .test_functions import ExpSmoothingForecasterFunction

    _HAS_SKTIME = True
except ImportError:
    _HAS_SKTIME = False

__all__ = [
    "GradientBoostingForecasterFunction",
    "RandomForestForecasterFunction",
]

if _HAS_SKTIME:
    __all__.append("ExpSmoothingForecasterFunction")
