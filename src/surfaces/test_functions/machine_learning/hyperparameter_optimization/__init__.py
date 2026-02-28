# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Hyperparameter Optimization Test Functions.

This module contains test functions for optimizing machine learning model hyperparameters
across different data types (tabular, image, time-series) and tasks (classification,
regression, forecasting).
"""

import importlib.util

_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


if _HAS_SKLEARN:
    from .image import (
        DeepCNNClassifierFunction,
        RandomForestImageClassifierFunction,
        SimpleCNNClassifierFunction,
        SVMImageClassifierFunction,
        XGBoostImageClassifierFunction,
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
        LightGBMClassifierFunction,
        LightGBMRegressorFunction,
        RandomForestClassifierFunction,
        RandomForestRegressorFunction,
        SVMClassifierFunction,
        SVMRegressorFunction,
        XGBoostClassifierFunction,
    )
    from .timeseries import (
        ExpSmoothingForecasterFunction,
        GradientBoostingForecasterFunction,
        KNNTSClassifierFunction,
        RandomForestForecasterFunction,
        RandomForestTSClassifierFunction,
        TSForestClassifierFunction,
    )

    __all__ = [
        # Tabular - Classification
        "DecisionTreeClassifierFunction",
        "GradientBoostingClassifierFunction",
        "KNeighborsClassifierFunction",
        "RandomForestClassifierFunction",
        "SVMClassifierFunction",
        "LightGBMClassifierFunction",
        "XGBoostClassifierFunction",
        # Tabular - Regression
        "DecisionTreeRegressorFunction",
        "GradientBoostingRegressorFunction",
        "KNeighborsRegressorFunction",
        "RandomForestRegressorFunction",
        "SVMRegressorFunction",
        "LightGBMRegressorFunction",
        # Time-series - Forecasting
        "GradientBoostingForecasterFunction",
        "RandomForestForecasterFunction",
        "ExpSmoothingForecasterFunction",
        # Time-series - Classification
        "RandomForestTSClassifierFunction",
        "KNNTSClassifierFunction",
        "TSForestClassifierFunction",
        # Image - Classification
        "SVMImageClassifierFunction",
        "RandomForestImageClassifierFunction",
        "SimpleCNNClassifierFunction",
        "DeepCNNClassifierFunction",
        "XGBoostImageClassifierFunction",
    ]

else:
    __all__ = []
