# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Registry of ML functions that support surrogate training.

This module defines which ML functions can have surrogates trained,
along with their fixed parameter grids (dataset, cv combinations).
"""

from typing import Any, Dict, List, Optional, Type

# Registry: function_name -> config
ML_SURROGATE_REGISTRY: Dict[str, Dict[str, Any]] = {}


DEFAULT_FIDELITY_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def register_ml_function(
    name: str,
    function_class: Type,
    fixed_params: Dict[str, List],
    hyperparams: List[str],
    fidelity_levels: Optional[List[float]] = None,
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
    fidelity_levels : list of float, optional
        Fidelity levels to evaluate during training. Defaults to
        [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]. Set to None or [1.0] to
        disable fidelity-aware training.
    """
    ML_SURROGATE_REGISTRY[name] = {
        "class": function_class,
        "fixed_params": fixed_params,
        "hyperparams": hyperparams,
        "fidelity_levels": fidelity_levels
        if fidelity_levels is not None
        else DEFAULT_FIDELITY_LEVELS,
    }


def get_registered_functions() -> List[str]:
    """Get list of registered function names."""
    _ensure_registered()
    return list(ML_SURROGATE_REGISTRY.keys())


def get_function_config(name: str) -> Dict[str, Any]:
    """Get configuration for a registered function."""
    _ensure_registered()
    if name not in ML_SURROGATE_REGISTRY:
        raise ValueError(f"Unknown function '{name}'. Available: {get_registered_functions()}")
    return ML_SURROGATE_REGISTRY[name]


def _ensure_registered():
    """Register all ML functions lazily on first access."""
    if ML_SURROGATE_REGISTRY:
        return  # Already registered

    from surfaces.test_functions.machine_learning import (
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

    # Dataset grids
    classification_datasets = ["digits", "iris", "wine", "breast_cancer", "covtype"]
    regression_datasets = ["diabetes", "california", "friedman1", "friedman2", "linear"]
    cv_options = [2, 3, 5, 10]

    register_ml_function(
        name="decision_tree_classifier",
        function_class=DecisionTreeClassifierFunction,
        fixed_params={"dataset": classification_datasets, "cv": cv_options},
        hyperparams=["max_depth", "min_samples_split", "min_samples_leaf"],
    )

    register_ml_function(
        name="gradient_boosting_classifier",
        function_class=GradientBoostingClassifierFunction,
        fixed_params={"dataset": classification_datasets, "cv": cv_options},
        hyperparams=["n_estimators", "max_depth", "learning_rate"],
    )

    register_ml_function(
        name="k_neighbors_classifier",
        function_class=KNeighborsClassifierFunction,
        fixed_params={"dataset": classification_datasets, "cv": cv_options},
        hyperparams=["n_neighbors", "algorithm"],
    )

    register_ml_function(
        name="random_forest_classifier",
        function_class=RandomForestClassifierFunction,
        fixed_params={"dataset": classification_datasets, "cv": cv_options},
        hyperparams=["n_estimators", "max_depth", "min_samples_split"],
    )

    register_ml_function(
        name="svm_classifier",
        function_class=SVMClassifierFunction,
        fixed_params={"dataset": classification_datasets, "cv": cv_options},
        hyperparams=["C", "kernel", "gamma"],
    )

    register_ml_function(
        name="decision_tree_regressor",
        function_class=DecisionTreeRegressorFunction,
        fixed_params={"dataset": regression_datasets, "cv": cv_options},
        hyperparams=["max_depth", "min_samples_split", "min_samples_leaf"],
    )

    register_ml_function(
        name="gradient_boosting_regressor",
        function_class=GradientBoostingRegressorFunction,
        fixed_params={"dataset": regression_datasets, "cv": cv_options},
        hyperparams=["n_estimators", "max_depth"],
    )

    register_ml_function(
        name="k_neighbors_regressor",
        function_class=KNeighborsRegressorFunction,
        fixed_params={"dataset": regression_datasets, "cv": cv_options},
        hyperparams=["n_neighbors", "algorithm"],
    )

    register_ml_function(
        name="random_forest_regressor",
        function_class=RandomForestRegressorFunction,
        fixed_params={"dataset": regression_datasets, "cv": cv_options},
        hyperparams=["n_estimators", "max_depth", "min_samples_split"],
    )

    register_ml_function(
        name="svm_regressor",
        function_class=SVMRegressorFunction,
        fixed_params={"dataset": regression_datasets, "cv": cv_options},
        hyperparams=["C", "kernel", "gamma"],
    )
