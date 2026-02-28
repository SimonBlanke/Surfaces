"""Time Series Forest Classifier test function using sktime."""

from typing import Any, Dict, List, Optional

from surfaces._dependencies import check_dependency
from surfaces.modifiers import BaseModifier

from .._base_ts_classification import BaseTSClassification
from ..datasets import DATASETS


class TSForestClassifierFunction(BaseTSClassification):
    """Time Series Forest Classifier test function using sktime.

    Uses sktime's TimeSeriesForestClassifier, an ensemble of decision trees
    built on random intervals of time series data.

    Parameters
    ----------
    dataset : str, default="gunpoint"
        Dataset to use. One of: "gunpoint", "ecg", "synthetic_ts".
    cv : int, default=5
        Number of cross-validation folds.
    use_surrogate : bool, default=False
        If True, use pre-trained surrogate for fast evaluation.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds.

    Examples
    --------
    >>> from surfaces.test_functions.machine_learning.timeseries import (
    ...     TSForestClassifierFunction
    ... )
    >>> func = TSForestClassifierFunction(dataset="gunpoint")
    >>> result = func({"n_estimators": 100, "min_interval": 3})
    """

    name = "Time Series Forest Classifier Function"
    _name_ = "ts_forest_classifier"

    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5]

    # Search space parameters
    para_names = ["n_estimators", "min_interval"]
    n_estimators_default = [50, 100, 150, 200, 250]
    min_interval_default = [3, 5, 7, 10, 15]

    def __init__(
        self,
        dataset: str = "gunpoint",
        cv: int = 5,
        objective: str = "maximize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks=None,
        catch_errors=None,
        use_surrogate: bool = False,
    ):
        check_dependency("sktime", "timeseries")

        if dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. " f"Available: {self.available_datasets}"
            )

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
        """Search space containing hyperparameters."""
        return {
            "n_estimators": self.n_estimators_default,
            "min_interval": self.min_interval_default,
        }

    def _ml_objective(self, params: Dict[str, Any]) -> float:
        from sklearn.model_selection import cross_val_score
        from sktime.classification.interval_based import TimeSeriesForestClassifier

        X_raw, y = self._dataset_loader()

        # sktime expects 3D array: (n_samples, n_channels, n_timepoints)
        X = X_raw.reshape(X_raw.shape[0], 1, X_raw.shape[1])

        model = TimeSeriesForestClassifier(
            n_estimators=params["n_estimators"],
            min_interval=params["min_interval"],
            random_state=42,
            n_jobs=-1,
        )

        scores = cross_val_score(model, X, y, cv=self.cv, scoring="accuracy")
        return scores.mean()

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add fixed parameters for surrogate prediction."""
        return {
            **params,
            "dataset": self.dataset,
            "cv": self.cv,
        }
