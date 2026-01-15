# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Hyperparameter Optimization Test Functions.

This module contains test functions for optimizing machine learning model hyperparameters
across different data types (tabular, image, time-series) and tasks (classification,
regression, forecasting).
"""

try:
    from sklearn.neighbors import KNeighborsClassifier  # noqa: F401

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


if _HAS_SKLEARN:
    # Tabular functions
    # Image functions (sklearn-based)
    from .image import (
        RandomForestImageClassifierFunction,
        SVMImageClassifierFunction,
    )
    from .tabular import (
        # Classification
        DecisionTreeClassifierFunction,
        # Regression
        DecisionTreeRegressorFunction,
        GradientBoostingClassifierFunction,
        GradientBoostingRegressorFunction,
        KNeighborsClassifierFunction,
        KNeighborsRegressorFunction,
        RandomForestClassifierFunction,
        RandomForestRegressorFunction,
        SVMClassifierFunction,
        SVMRegressorFunction,
    )

    # Time-series functions (sklearn-based)
    from .timeseries import (
        # Forecasting
        GradientBoostingForecasterFunction,
        KNNTSClassifierFunction,
        RandomForestForecasterFunction,
        # Classification
        RandomForestTSClassifierFunction,
    )

    __all__ = [
        # Tabular - Classification
        "DecisionTreeClassifierFunction",
        "GradientBoostingClassifierFunction",
        "KNeighborsClassifierFunction",
        "RandomForestClassifierFunction",
        "SVMClassifierFunction",
        # Tabular - Regression
        "DecisionTreeRegressorFunction",
        "GradientBoostingRegressorFunction",
        "KNeighborsRegressorFunction",
        "RandomForestRegressorFunction",
        "SVMRegressorFunction",
        # Time-series - Forecasting
        "GradientBoostingForecasterFunction",
        "RandomForestForecasterFunction",
        # Time-series - Classification
        "RandomForestTSClassifierFunction",
        "KNNTSClassifierFunction",
        # Image - Classification (sklearn)
        "SVMImageClassifierFunction",
        "RandomForestImageClassifierFunction",
    ]

    # sktime-based time-series functions (require sktime)
    try:
        from .timeseries import ExpSmoothingForecasterFunction, TSForestClassifierFunction

        __all__.extend(
            [
                "ExpSmoothingForecasterFunction",
                "TSForestClassifierFunction",
            ]
        )
        _HAS_SKTIME = True
    except ImportError:
        _HAS_SKTIME = False

    # CNN image classifiers (require tensorflow)
    try:
        from .image import DeepCNNClassifierFunction, SimpleCNNClassifierFunction

        __all__.extend(
            [
                "SimpleCNNClassifierFunction",
                "DeepCNNClassifierFunction",
            ]
        )
        _HAS_TENSORFLOW = True
    except ImportError:
        _HAS_TENSORFLOW = False

    # XGBoost image classifier (requires xgboost)
    try:
        from .image import XGBoostImageClassifierFunction

        __all__.append("XGBoostImageClassifierFunction")
        _HAS_XGBOOST = True
    except ImportError:
        _HAS_XGBOOST = False

else:
    __all__ = []
