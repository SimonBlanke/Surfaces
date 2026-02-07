# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

try:
    from sklearn.neighbors import KNeighborsClassifier  # noqa: F401

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _check_sklearn():
    """Check if scikit-learn is available."""
    if not _HAS_SKLEARN:
        raise ImportError(
            "Machine learning functions require scikit-learn. "
            "Install with: pip install surfaces[ml]"
        )


if _HAS_SKLEARN:
    # Import from new hyperparameter_optimization module
    # Import from ensemble_optimization module
    from .ensemble_optimization import (
        StackingEnsembleFunction,
        VotingEnsembleFunction,
        WeightedAveragingFunction,
    )

    # Import from feature_engineering module
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
        GradientBoostingClassifierFunction,
        # Time-series
        GradientBoostingForecasterFunction,
        GradientBoostingRegressorFunction,
        KNeighborsClassifierFunction,
        KNeighborsRegressorFunction,
        KNNTSClassifierFunction,
        RandomForestClassifierFunction,
        RandomForestForecasterFunction,
        # Image
        RandomForestImageClassifierFunction,
        RandomForestRegressorFunction,
        RandomForestTSClassifierFunction,
        SVMClassifierFunction,
        SVMImageClassifierFunction,
        SVMRegressorFunction,
    )

    # Import from llm_optimization module (no additional deps needed for mock mode)
    from .llm_optimization import (
        PromptEngineeringFunction,
        RAGOptimizationFunction,
    )

    # Import from pipelines module
    from .pipelines import (
        ClassificationPipelineFunction,
        FeatureEngineeringPipelineFunction,
        RegressionPipelineFunction,
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
        # Feature Engineering
        "MutualInfoFeatureSelectionFunction",
        "PolynomialFeatureTransformationFunction",
        "FeatureScalingPipelineFunction",
        # Ensemble Optimization
        "VotingEnsembleFunction",
        "StackingEnsembleFunction",
        "WeightedAveragingFunction",
        # Pipelines
        "ClassificationPipelineFunction",
        "RegressionPipelineFunction",
        "FeatureEngineeringPipelineFunction",
        # LLM Optimization (Mock Mode)
        "PromptEngineeringFunction",
        "RAGOptimizationFunction",
    ]

    machine_learning_functions = [
        # Tabular - Classification
        DecisionTreeClassifierFunction,
        GradientBoostingClassifierFunction,
        KNeighborsClassifierFunction,
        RandomForestClassifierFunction,
        SVMClassifierFunction,
        # Tabular - Regression
        DecisionTreeRegressorFunction,
        GradientBoostingRegressorFunction,
        KNeighborsRegressorFunction,
        RandomForestRegressorFunction,
        SVMRegressorFunction,
        # Time-series - Forecasting
        GradientBoostingForecasterFunction,
        RandomForestForecasterFunction,
        # Time-series - Classification
        RandomForestTSClassifierFunction,
        KNNTSClassifierFunction,
        # Image - Classification (sklearn)
        SVMImageClassifierFunction,
        RandomForestImageClassifierFunction,
        # Feature Engineering
        MutualInfoFeatureSelectionFunction,
        PolynomialFeatureTransformationFunction,
        FeatureScalingPipelineFunction,
        # Ensemble Optimization
        VotingEnsembleFunction,
        StackingEnsembleFunction,
        WeightedAveragingFunction,
        # Pipelines
        ClassificationPipelineFunction,
        RegressionPipelineFunction,
        FeatureEngineeringPipelineFunction,
        # LLM Optimization (Mock Mode)
        PromptEngineeringFunction,
        RAGOptimizationFunction,
    ]

    # sktime-based time-series functions (require sktime)
    try:
        from .hyperparameter_optimization.timeseries import (
            ExpSmoothingForecasterFunction,
            TSForestClassifierFunction,
        )

        __all__.extend(
            [
                "ExpSmoothingForecasterFunction",
                "TSForestClassifierFunction",
            ]
        )
        machine_learning_functions.extend(
            [
                ExpSmoothingForecasterFunction,
                TSForestClassifierFunction,
            ]
        )
        _HAS_SKTIME = True
    except ImportError:
        _HAS_SKTIME = False

    # CNN image classifiers (require tensorflow)
    try:
        from .hyperparameter_optimization.image import (
            DeepCNNClassifierFunction,
            SimpleCNNClassifierFunction,
        )

        __all__.extend(
            [
                "SimpleCNNClassifierFunction",
                "DeepCNNClassifierFunction",
            ]
        )
        machine_learning_functions.extend(
            [
                SimpleCNNClassifierFunction,
                DeepCNNClassifierFunction,
            ]
        )
        _HAS_TENSORFLOW = True
    except ImportError:
        _HAS_TENSORFLOW = False

    # XGBoost image classifier (requires xgboost)
    try:
        from .hyperparameter_optimization.image import XGBoostImageClassifierFunction

        __all__.append("XGBoostImageClassifierFunction")
        machine_learning_functions.append(XGBoostImageClassifierFunction)
        _HAS_XGBOOST = True
    except ImportError:
        _HAS_XGBOOST = False

    # Neural Architecture Search (require PyTorch and/or TensorFlow)
    try:
        from .neural_architecture_search import (
            CNNKerasNASFunction,
            MLPPyTorchNASFunction,
        )

        __all__.extend(
            [
                "MLPPyTorchNASFunction",
                "CNNKerasNASFunction",
            ]
        )
        machine_learning_functions.extend(
            [
                MLPPyTorchNASFunction,
                CNNKerasNASFunction,
            ]
        )
        _HAS_NAS = True
    except ImportError:
        _HAS_NAS = False

    # Transfer Learning (requires TensorFlow)
    try:
        from .transfer_learning import SimpleTransferLearningFunction

        __all__.append("SimpleTransferLearningFunction")
        machine_learning_functions.append(SimpleTransferLearningFunction)
        _HAS_TRANSFER_LEARNING = True
    except ImportError:
        _HAS_TRANSFER_LEARNING = False

    # Data Augmentation (requires TensorFlow)
    try:
        from .data_augmentation import ImageAugmentationFunction

        __all__.append("ImageAugmentationFunction")
        machine_learning_functions.append(ImageAugmentationFunction)
        _HAS_DATA_AUGMENTATION = True
    except ImportError:
        _HAS_DATA_AUGMENTATION = False

    # Reinforcement Learning (requires gymnasium and PyTorch)
    try:
        from .reinforcement_learning import DQNCartPoleFunction

        __all__.append("DQNCartPoleFunction")
        machine_learning_functions.append(DQNCartPoleFunction)
        _HAS_RL = True
    except ImportError:
        _HAS_RL = False

else:
    __all__ = []
    machine_learning_functions = []
