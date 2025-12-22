# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Registry of ML functions that support surrogate training.

This module defines which ML functions can have surrogates trained,
along with their fixed parameter grids (dataset, cv combinations).
"""

from typing import Any, Dict, List, Type

# Registry: function_name -> config
ML_SURROGATE_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_ml_function(
    name: str,
    function_class: Type,
    fixed_params: Dict[str, List],
    hyperparams: List[str],
):
    """Register an ML function for surrogate training.

    Parameters
    ----------
    name : str
        Unique identifier (e.g., "k_neighbors_classifier").
    function_class : Type
        The function class (e.g., KNeighborsClassifierFunction).
    fixed_params : dict
        Grid of fixed parameters to iterate over during training.
        Example: {"dataset": ["iris", "digits"], "cv": [2, 3, 5, 10]}
    hyperparams : list
        Names of hyperparameters in the search space.
    """
    ML_SURROGATE_REGISTRY[name] = {
        "class": function_class,
        "fixed_params": fixed_params,
        "hyperparams": hyperparams,
    }


def get_registered_functions() -> List[str]:
    """Get list of registered function names."""
    _ensure_registered()
    return list(ML_SURROGATE_REGISTRY.keys())


def get_function_config(name: str) -> Dict[str, Any]:
    """Get configuration for a registered function."""
    _ensure_registered()
    if name not in ML_SURROGATE_REGISTRY:
        raise ValueError(
            f"Unknown function '{name}'. "
            f"Available: {get_registered_functions()}"
        )
    return ML_SURROGATE_REGISTRY[name]


# ============================================================================
# Register ML functions (lazy to avoid circular imports)
# ============================================================================

def _ensure_registered():
    """Register all ML functions lazily on first access."""
    if ML_SURROGATE_REGISTRY:
        return  # Already registered

    from surfaces.test_functions import (
        KNeighborsClassifierFunction,
        KNeighborsRegressorFunction,
        GradientBoostingRegressorFunction,
    )

    # Classification functions
    register_ml_function(
        name="k_neighbors_classifier",
        function_class=KNeighborsClassifierFunction,
        fixed_params={
            "dataset": ["digits", "iris", "wine"],
            "cv": [2, 3, 5, 10],
        },
        hyperparams=["n_neighbors", "algorithm"],
    )

    # Regression functions
    register_ml_function(
        name="k_neighbors_regressor",
        function_class=KNeighborsRegressorFunction,
        fixed_params={
            "dataset": ["diabetes", "california"],
            "cv": [2, 3, 5, 10],
        },
        hyperparams=["n_neighbors", "algorithm"],
    )

    register_ml_function(
        name="gradient_boosting_regressor",
        function_class=GradientBoostingRegressorFunction,
        fixed_params={
            "dataset": ["diabetes", "california"],
            "cv": [2, 3, 5, 10],
        },
        hyperparams=["n_estimators", "max_depth"],
    )
