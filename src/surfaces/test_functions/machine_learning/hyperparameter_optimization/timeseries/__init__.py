# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .classification import (
    KNNTSClassifierFunction,
    RandomForestTSClassifierFunction,
)
from .forecasting import (
    GradientBoostingForecasterFunction,
    RandomForestForecasterFunction,
)

# sktime-based functions (require sktime)
try:
    from .classification import TSForestClassifierFunction
    from .forecasting import ExpSmoothingForecasterFunction

    _HAS_SKTIME = True
except ImportError:
    _HAS_SKTIME = False

__all__ = [
    # Forecasting
    "GradientBoostingForecasterFunction",
    "RandomForestForecasterFunction",
    # Classification
    "RandomForestTSClassifierFunction",
    "KNNTSClassifierFunction",
]

if _HAS_SKTIME:
    __all__.extend(
        [
            "ExpSmoothingForecasterFunction",
            "TSForestClassifierFunction",
        ]
    )
