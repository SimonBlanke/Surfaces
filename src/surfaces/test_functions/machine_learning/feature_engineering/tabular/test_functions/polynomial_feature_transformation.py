"""Polynomial Feature Transformation test function."""

from typing import Any, Callable, Dict, List, Optional, Union

from surfaces.modifiers import BaseModifier
from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.regression.datasets import (
    DATASETS,
)

from .._base_tabular_feature_engineering import BaseTabularFeatureEngineering


class PolynomialFeatureTransformationFunction(BaseTabularFeatureEngineering):
    """Polynomial Feature Transformation test function.

    Optimizes polynomial feature generation for regression tasks. Evaluates
    different polynomial degrees and interaction settings to improve model
    performance.

    Parameters
    ----------
    dataset : str, default="diabetes"
        Dataset to use for evaluation. One of: "diabetes", "california_housing".
    cv : int, default=5
        Number of cross-validation folds.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> from surfaces.test_functions import PolynomialFeatureTransformationFunction
    >>> func = PolynomialFeatureTransformationFunction(dataset="diabetes", cv=5)
    >>> func.search_space
    {'degree': [1, 2, 3], 'interaction_only': [True, False], ...}
    >>> result = func({"degree": 2, "interaction_only": False, "include_bias": True})
    """

    name = "Polynomial Feature Transformation"

    available_datasets = ["diabetes", "california_housing"]
    available_cv = [2, 3, 5, 10]

    para_names = ["degree", "interaction_only", "include_bias"]
    degree_default = [1, 2, 3]
    interaction_only_default = [True, False]
    include_bias_default = [True, False]

    def __init__(
        self,
        dataset: str = "diabetes",
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

    def _default_search_space(self) -> Dict[str, Any]:
        """Search space for polynomial feature transformation."""
        return {
            "degree": self.degree_default,
            "interaction_only": self.interaction_only_default,
            "include_bias": self.include_bias_default,
        }

    def _ml_objective(self, params: Dict[str, Any]) -> float:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import PolynomialFeatures

        X, y = self._dataset_loader()

        poly = PolynomialFeatures(
            degree=params["degree"],
            interaction_only=params["interaction_only"],
            include_bias=params["include_bias"],
        )
        X_poly = poly.fit_transform(X)

        model = Ridge(alpha=1.0, random_state=42)

        scores = cross_val_score(model, X_poly, y, cv=self.cv, scoring="r2")
        return scores.mean()
