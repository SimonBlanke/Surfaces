"""Feature Engineering Pipeline test function."""

from typing import Any, Callable, Dict, List, Optional, Union

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from surfaces.modifiers import BaseModifier
from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.classification.datasets import (
    DATASETS,
)

from .._base_tabular_pipeline import BaseTabularPipeline


class FeatureEngineeringPipelineFunction(BaseTabularPipeline):
    """Feature Engineering Pipeline test function.

    Optimizes a complex feature engineering pipeline that combines multiple
    transformation steps: polynomial features, feature selection, dimensionality
    reduction, and scaling. This represents a comprehensive preprocessing pipeline.

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
    >>> from surfaces.test_functions import FeatureEngineeringPipelineFunction
    >>> func = FeatureEngineeringPipelineFunction(dataset="digits", cv=5)
    >>> func.search_space
    {'use_poly': [True, False], 'use_selection': [True, False], ...}
    >>> result = func({"use_poly": False, "use_selection": True, "selection_k": 30,
    ...                "use_pca": True, "pca_components": 20})
    """

    name = "Feature Engineering Pipeline"
    _name_ = "feature_engineering_pipeline"
    __name__ = "FeatureEngineeringPipelineFunction"

    available_datasets = ["digits", "iris", "wine", "breast_cancer"]
    available_cv = [2, 3, 5, 10]

    para_names = ["use_poly", "use_selection", "selection_k", "use_pca", "pca_components"]
    use_poly_default = [True, False]
    use_selection_default = [True, False]
    use_pca_default = [True, False]

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

        # Load dataset to determine feature counts
        X, y = self._dataset_loader()
        n_features = X.shape[1]

        # Create dynamic search space based on dataset
        self.selection_k_default = list(range(5, min(n_features, 50), 5))
        self.pca_components_default = list(range(5, min(n_features, 40), 5))

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
        """Search space for feature engineering pipeline optimization."""
        return {
            "use_poly": self.use_poly_default,
            "use_selection": self.use_selection_default,
            "selection_k": self.selection_k_default,
            "use_pca": self.use_pca_default,
            "pca_components": self.pca_components_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function for feature engineering pipeline."""
        X, y = self._dataset_loader()
        cv = self.cv
        n_features = X.shape[1]

        def objective_function(params: Dict[str, Any]) -> float:
            # Build pipeline steps
            steps = []

            # 1. Polynomial features (optional)
            # Note: This can greatly increase dimensionality
            if params["use_poly"]:
                steps.append(
                    (
                        "poly",
                        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                    )
                )

            # 2. Feature selection (optional)
            if params["use_selection"]:
                # Determine k based on current feature count
                k = min(params["selection_k"], n_features)
                steps.append(("selection", SelectKBest(f_classif, k=k)))

            # 3. PCA dimensionality reduction (optional)
            if params["use_pca"]:
                # Determine n_components based on current feature count
                n_components = min(params["pca_components"], n_features)
                steps.append(("pca", PCA(n_components=n_components, random_state=42)))

            # 4. Scaling (always apply before model)
            steps.append(("scaler", StandardScaler()))

            # 5. Model (fixed for this function)
            steps.append(("model", RandomForestClassifier(n_estimators=50, random_state=42)))

            # Need at least some transformation
            if len(steps) <= 2:  # Only scaler + model
                # All transformations disabled, return baseline
                pass

            # Create and evaluate pipeline
            pipeline = Pipeline(steps)

            try:
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
                return scores.mean()
            except ValueError:
                # Invalid configuration (e.g., too many PCA components)
                return 0.0

        self.pure_objective_function = objective_function
