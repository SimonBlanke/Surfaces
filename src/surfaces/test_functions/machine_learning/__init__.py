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
        GradientBoostingRegressorFunction,
        KNeighborsClassifierFunction,
        KNeighborsRegressorFunction,
    )

    __all__ = [
        "KNeighborsClassifierFunction",
        "KNeighborsRegressorFunction",
        "GradientBoostingRegressorFunction",
    ]

    machine_learning_functions = [
        KNeighborsClassifierFunction,
        GradientBoostingRegressorFunction,
        KNeighborsRegressorFunction,
    ]
else:
    __all__ = []
    machine_learning_functions = []
