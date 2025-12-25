from typing import Any, Dict

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Time Series Forest Classifier test function using sktime."""

from .._base_ts_classification import BaseTSClassification
from ..datasets import DATASETS


def _check_sktime():
    """Check if sktime is available."""
    try:
        import sktime  # noqa: F401

        return True
    except ImportError:
        raise ImportError(
            "Time Series Forest classifier requires sktime. "
            "Install with: pip install surfaces[timeseries]"
        )


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
    __name__ = "TSForestClassifierFunction"

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
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks=None,
        catch_errors=None,
        noise=None,
        use_surrogate: bool = False,
    ):
        _check_sktime()

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
            sleep=sleep,
            memory=memory,
            collect_data=collect_data,
            callbacks=callbacks,
            catch_errors=catch_errors,
            noise=noise,
            use_surrogate=use_surrogate,
        )

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space containing hyperparameters."""
        return {
            "n_estimators": self.n_estimators_default,
            "min_interval": self.min_interval_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function with fixed dataset and cv."""
        from sklearn.model_selection import cross_val_score
        from sktime.classification.interval_based import TimeSeriesForestClassifier

        X_raw, y = self._dataset_loader()
        cv = self.cv

        # sktime expects 3D array: (n_samples, n_channels, n_timepoints)
        # Our data is 2D: (n_samples, n_timepoints)
        # Reshape to add channel dimension
        X = X_raw.reshape(X_raw.shape[0], 1, X_raw.shape[1])

        def ts_forest_classifier(params: Dict[str, Any]) -> float:
            model = TimeSeriesForestClassifier(
                n_estimators=params["n_estimators"],
                min_interval=params["min_interval"],
                random_state=42,
                n_jobs=-1,
            )

            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        self.pure_objective_function = ts_forest_classifier

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add fixed parameters for surrogate prediction."""
        return {
            **params,
            "dataset": self.dataset,
            "cv": self.cv,
        }
