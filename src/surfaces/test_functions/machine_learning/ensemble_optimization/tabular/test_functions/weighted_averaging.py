"""Weighted Averaging Ensemble test function."""

from typing import Any, Callable, Dict, List, Optional, Union

from surfaces.modifiers import BaseModifier
from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.regression.datasets import (
    DATASETS,
)

from .._base_tabular_ensemble import BaseTabularEnsemble


class WeightedAveragingFunction(BaseTabularEnsemble):
    """Weighted Averaging Ensemble test function.

    Optimizes weighted averaging of multiple regression models. The function
    evaluates different weight combinations to find the optimal linear combination
    of base model predictions.

    Parameters
    ----------
    dataset : str, default="diabetes"
        Dataset to use for evaluation. One of: "diabetes", "california".
    cv : int, default=5
        Number of cross-validation folds.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> from surfaces.test_functions import WeightedAveragingFunction
    >>> func = WeightedAveragingFunction(dataset="diabetes", cv=5)
    >>> func.search_space
    {'dt_weight': [0.0, 0.25, 0.5, 0.75, 1.0], ...}
    >>> result = func({"dt_weight": 0.25, "rf_weight": 0.5, "gb_weight": 0.25})
    """

    name = "Weighted Averaging Ensemble"
    _name_ = "weighted_averaging"

    available_datasets = ["diabetes", "california"]
    available_cv = [2, 3, 5, 10]

    para_names = ["dt_weight", "rf_weight", "gb_weight"]
    dt_weight_default = [0.0, 0.25, 0.5, 0.75, 1.0]
    rf_weight_default = [0.0, 0.25, 0.5, 0.75, 1.0]
    gb_weight_default = [0.0, 0.25, 0.5, 0.75, 1.0]

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

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space for weighted averaging optimization."""
        return {
            "dt_weight": self.dt_weight_default,
            "rf_weight": self.rf_weight_default,
            "gb_weight": self.gb_weight_default,
        }

    def _ml_objective(self, params: Dict[str, Any]) -> float:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.metrics import r2_score
        from sklearn.model_selection import cross_val_predict
        from sklearn.tree import DecisionTreeRegressor

        X, y = self._dataset_loader()

        dt_model = DecisionTreeRegressor(random_state=42)
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        gb_model = GradientBoostingRegressor(n_estimators=50, random_state=42)

        dt_preds = cross_val_predict(dt_model, X, y, cv=self.cv)
        rf_preds = cross_val_predict(rf_model, X, y, cv=self.cv)
        gb_preds = cross_val_predict(gb_model, X, y, cv=self.cv)

        dt_weight = params["dt_weight"]
        rf_weight = params["rf_weight"]
        gb_weight = params["gb_weight"]

        total_weight = dt_weight + rf_weight + gb_weight
        if total_weight == 0.0:
            return 0.0

        dt_weight_norm = dt_weight / total_weight
        rf_weight_norm = rf_weight / total_weight
        gb_weight_norm = gb_weight / total_weight

        weighted_preds = (
            dt_weight_norm * dt_preds + rf_weight_norm * rf_preds + gb_weight_norm * gb_preds
        )

        score = r2_score(y, weighted_preds)
        return score
