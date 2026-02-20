# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import importlib.util

_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def _check_sklearn():
    """Check if scikit-learn is available."""
    if not _HAS_SKLEARN:
        raise ImportError(
            "Machine learning functions require scikit-learn. "
            "Install with: pip install surfaces[ml]"
        )


if _HAS_SKLEARN:
    from .hyperparameter_optimization import (
        # Tabular - Classification
        DecisionTreeClassifierFunction,
        # Tabular - Regression
        DecisionTreeRegressorFunction,
        # Image - Classification
        DeepCNNClassifierFunction,
        # Time-series - Forecasting
        ExpSmoothingForecasterFunction,
        GradientBoostingClassifierFunction,
        GradientBoostingForecasterFunction,
        GradientBoostingRegressorFunction,
        KNeighborsClassifierFunction,
        KNeighborsRegressorFunction,
        # Time-series - Classification
        KNNTSClassifierFunction,
        LightGBMClassifierFunction,
        LightGBMRegressorFunction,
        RandomForestClassifierFunction,
        RandomForestForecasterFunction,
        RandomForestImageClassifierFunction,
        RandomForestRegressorFunction,
        RandomForestTSClassifierFunction,
        SimpleCNNClassifierFunction,
        SVMClassifierFunction,
        SVMImageClassifierFunction,
        SVMRegressorFunction,
        TSForestClassifierFunction,
        XGBoostImageClassifierFunction,
    )

    __all__ = [
        # Tabular - Classification
        "DecisionTreeClassifierFunction",
        "GradientBoostingClassifierFunction",
        "KNeighborsClassifierFunction",
        "RandomForestClassifierFunction",
        "SVMClassifierFunction",
        "LightGBMClassifierFunction",
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

    machine_learning_functions = [
        # Tabular - Classification
        DecisionTreeClassifierFunction,
        GradientBoostingClassifierFunction,
        KNeighborsClassifierFunction,
        RandomForestClassifierFunction,
        SVMClassifierFunction,
        LightGBMClassifierFunction,
        # Tabular - Regression
        DecisionTreeRegressorFunction,
        GradientBoostingRegressorFunction,
        KNeighborsRegressorFunction,
        RandomForestRegressorFunction,
        SVMRegressorFunction,
        LightGBMRegressorFunction,
        # Time-series - Forecasting
        GradientBoostingForecasterFunction,
        RandomForestForecasterFunction,
        ExpSmoothingForecasterFunction,
        # Time-series - Classification
        RandomForestTSClassifierFunction,
        KNNTSClassifierFunction,
        TSForestClassifierFunction,
        # Image - Classification
        SVMImageClassifierFunction,
        RandomForestImageClassifierFunction,
        SimpleCNNClassifierFunction,
        DeepCNNClassifierFunction,
        XGBoostImageClassifierFunction,
    ]

else:
    __all__ = []
    machine_learning_functions = []
