# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import importlib.util

_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


if _HAS_SKLEARN:
    from .ensemble_optimization import (
        StackingEnsembleFunction,
        VotingEnsembleFunction,
        WeightedAveragingFunction,
    )
    from .feature_engineering import (
        FeatureScalingPipelineFunction,
        MutualInfoFeatureSelectionFunction,
        PolynomialFeatureTransformationFunction,
    )
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
        XGBoostClassifierFunction,
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
        # Ensemble Optimization
        "StackingEnsembleFunction",
        "VotingEnsembleFunction",
        "WeightedAveragingFunction",
        # Feature Engineering
        "FeatureScalingPipelineFunction",
        "MutualInfoFeatureSelectionFunction",
        "PolynomialFeatureTransformationFunction",
    ]

    machine_learning_functions = [
        # Tabular - Classification
        DecisionTreeClassifierFunction,
        GradientBoostingClassifierFunction,
        KNeighborsClassifierFunction,
        RandomForestClassifierFunction,
        SVMClassifierFunction,
        LightGBMClassifierFunction,
        XGBoostClassifierFunction,
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
        # Ensemble Optimization
        StackingEnsembleFunction,
        VotingEnsembleFunction,
        WeightedAveragingFunction,
        # Feature Engineering
        FeatureScalingPipelineFunction,
        MutualInfoFeatureSelectionFunction,
        PolynomialFeatureTransformationFunction,
    ]

else:
    __all__ = []
    machine_learning_functions = []
