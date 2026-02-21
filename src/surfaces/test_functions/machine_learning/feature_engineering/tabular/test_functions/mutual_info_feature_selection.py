"""Mutual Information Feature Selection test function."""

from typing import Any, Callable, Dict, List, Optional, Union

from surfaces.modifiers import BaseModifier
from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.classification.datasets import (
    DATASETS,
)

from .._base_tabular_feature_engineering import BaseTabularFeatureEngineering


class MutualInfoFeatureSelectionFunction(BaseTabularFeatureEngineering):
    """Mutual Information Feature Selection test function.

    Optimizes feature selection using mutual information criterion combined
    with different classification models. Evaluates how well selected features
    perform with various classifiers.

    Parameters
    ----------
    dataset : str, default="digits"
        Dataset to use for evaluation. One of: "digits", "iris", "wine", "breast_cancer".
    cv : int, default=5
        Number of cross-validation folds.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> from surfaces.test_functions import MutualInfoFeatureSelectionFunction
    >>> func = MutualInfoFeatureSelectionFunction(dataset="digits", cv=5)
    >>> func.search_space
    {'n_features': [5, 10, 15, ...], 'model_type': ['dt', 'rf', 'gb']}
    >>> result = func({"n_features": 20, "model_type": "rf"})
    """

    name = "Mutual Information Feature Selection"
    _name_ = "mutual_info_feature_selection"

    available_datasets = ["digits", "iris", "wine", "breast_cancer"]
    available_cv = [2, 3, 5, 10]

    para_names = ["n_features", "model_type"]

    def __init__(
        self,
        dataset: str = "digits",
        cv: int = 5,
        objective: str = "maximize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        use_surrogate: bool = False,
    ):
        if dataset not in self.available_datasets:
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {self.available_datasets}")

        if cv not in self.available_cv:
            raise ValueError(f"Invalid cv={cv}. Available: {self.available_cv}")

        self.dataset = dataset
        self.cv = cv
        self._dataset_loader = DATASETS[dataset]

        # Load dataset to determine n_features range
        X, y = self._dataset_loader()
        n_features_total = X.shape[1]

        # Create search space based on dataset
        self.n_features_default = list(range(2, min(n_features_total, 50), 2))
        self.model_type_default = ["dt", "rf", "gb"]

        super().__init__(
            objective=objective,
            modifiers=modifiers,
            memory=memory,
            collect_data=collect_data,
            callbacks=callbacks,
            catch_errors=catch_errors,
            use_surrogate=use_surrogate,
        )

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space for feature selection optimization."""
        return {
            "n_features": self.n_features_default,
            "model_type": self.model_type_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function for feature selection."""
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier

        X, y = self._dataset_loader()
        cv = self.cv

        def objective_function(params: Dict[str, Any]) -> float:
            # Select features using mutual information
            n_features = params["n_features"]
            selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
            X_selected = selector.fit_transform(X, y)

            # Train model on selected features
            model_type = params["model_type"]
            if model_type == "dt":
                model = DecisionTreeClassifier(random_state=42)
            elif model_type == "rf":
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            elif model_type == "gb":
                model = GradientBoostingClassifier(n_estimators=50, random_state=42)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # Evaluate
            scores = cross_val_score(model, X_selected, y, cv=cv, scoring="accuracy")
            return scores.mean()

        self.pure_objective_function = objective_function
