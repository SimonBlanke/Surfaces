# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .gradient_boosting_forecaster import GradientBoostingForecasterFunction
from .random_forest_forecaster import RandomForestForecasterFunction

# sktime-based forecasters (require sktime)
try:
    from .exp_smoothing_forecaster import ExpSmoothingForecasterFunction

    _HAS_SKTIME = True
except ImportError:
    _HAS_SKTIME = False

__all__ = [
    "GradientBoostingForecasterFunction",
    "RandomForestForecasterFunction",
]

if _HAS_SKTIME:
    __all__.append("ExpSmoothingForecasterFunction")
