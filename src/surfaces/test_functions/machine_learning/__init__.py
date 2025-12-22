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
    from .tabular import (
        # Classification
        DecisionTreeClassifierFunction,
        GradientBoostingClassifierFunction,
        KNeighborsClassifierFunction,
        RandomForestClassifierFunction,
        SVMClassifierFunction,
        # Regression
        DecisionTreeRegressorFunction,
        GradientBoostingRegressorFunction,
        KNeighborsRegressorFunction,
        RandomForestRegressorFunction,
        SVMRegressorFunction,
    )

    __all__ = [
        # Classification
        "DecisionTreeClassifierFunction",
        "GradientBoostingClassifierFunction",
        "KNeighborsClassifierFunction",
        "RandomForestClassifierFunction",
        "SVMClassifierFunction",
        # Regression
        "DecisionTreeRegressorFunction",
        "GradientBoostingRegressorFunction",
        "KNeighborsRegressorFunction",
        "RandomForestRegressorFunction",
        "SVMRegressorFunction",
    ]

    machine_learning_functions = [
        # Classification
        DecisionTreeClassifierFunction,
        GradientBoostingClassifierFunction,
        KNeighborsClassifierFunction,
        RandomForestClassifierFunction,
        SVMClassifierFunction,
        # Regression
        DecisionTreeRegressorFunction,
        GradientBoostingRegressorFunction,
        KNeighborsRegressorFunction,
        RandomForestRegressorFunction,
        SVMRegressorFunction,
    ]
else:
    __all__ = []
    machine_learning_functions = []
