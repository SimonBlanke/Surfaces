"""Feature Scaling Pipeline test function."""

from typing import Any, Callable, Dict, List, Optional, Union

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC

from surfaces.modifiers import BaseModifier
from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.classification.datasets import (
    DATASETS,
)

from .._base_tabular_feature_engineering import BaseTabularFeatureEngineering


class FeatureScalingPipelineFunction(BaseTabularFeatureEngineering):
    """Feature Scaling Pipeline test function.

    Optimizes the combination of scaling method and classification model.
    Different models benefit from different scaling strategies, and this
    function helps identify the best combination.

    Parameters
    ----------
    dataset : str, default="wine"
        Dataset to use for evaluation. One of: "digits", "iris", "wine", "breast_cancer".
    cv : int, default=5
        Number of cross-validation folds.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> from surfaces.test_functions import FeatureScalingPipelineFunction
    >>> func = FeatureScalingPipelineFunction(dataset="wine", cv=5)
    >>> func.search_space
    {'scaler': ['standard', 'minmax', 'robust', 'none'], 'model_type': ['svm', 'gb']}
    >>> result = func({"scaler": "standard", "model_type": "svm"})
    """

    name = "Feature Scaling Pipeline"
    _name_ = "feature_scaling_pipeline"
    __name__ = "FeatureScalingPipelineFunction"

    available_datasets = ["digits", "iris", "wine", "breast_cancer"]
    available_cv = [2, 3, 5, 10]

    para_names = ["scaler", "model_type"]
    scaler_default = ["standard", "minmax", "robust", "none"]
    model_type_default = ["svm", "gb"]

    def __init__(
        self,
        dataset: str = "wine",
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
        """Search space for feature scaling pipeline."""
        return {
            "scaler": self.scaler_default,
            "model_type": self.model_type_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function for feature scaling pipeline."""
        X, y = self._dataset_loader()
        cv = self.cv

        def objective_function(params: Dict[str, Any]) -> float:
            # Apply scaling
            scaler_type = params["scaler"]
            if scaler_type == "standard":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            elif scaler_type == "minmax":
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
            elif scaler_type == "robust":
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
            elif scaler_type == "none":
                X_scaled = X
            else:
                raise ValueError(f"Unknown scaler: {scaler_type}")

            # Train model on scaled features
            model_type = params["model_type"]
            if model_type == "svm":
                model = SVC(kernel="rbf", random_state=42)
            elif model_type == "gb":
                model = GradientBoostingClassifier(n_estimators=50, random_state=42)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # Evaluate
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
            return scores.mean()

        self.pure_objective_function = objective_function
