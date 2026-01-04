"""Random Forest Time-Series Classifier test function using feature extraction."""

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from .._base_ts_classification import BaseTSClassification
from ..datasets import DATASETS


def extract_ts_features(X):
    """Extract statistical features from time-series data.

    Parameters
    ----------
    X : ndarray
        Time-series data of shape (n_samples, n_timepoints).

    Returns
    -------
    features : ndarray
        Feature matrix of shape (n_samples, n_features).
    """
    features = []

    for ts in X:
        ts_features = [
            np.mean(ts),
            np.std(ts),
            np.min(ts),
            np.max(ts),
            np.percentile(ts, 25),
            np.percentile(ts, 75),
            np.mean(np.diff(ts)),  # Average change
            np.std(np.diff(ts)),  # Variability of changes
            np.sum(np.abs(np.diff(ts))),  # Total variation
            np.argmax(ts) / len(ts),  # Relative position of max
            np.argmin(ts) / len(ts),  # Relative position of min
            len(np.where(np.diff(np.sign(ts)))[0]),  # Zero crossings
        ]
        features.append(ts_features)

    return np.array(features)


class RandomForestTSClassifierFunction(BaseTSClassification):
    """Random Forest Time-Series Classifier using feature extraction.

    Extracts statistical features from time-series and uses Random Forest
    for classification. A sklearn-only approach without sktime dependency.

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
    ...     RandomForestTSClassifierFunction
    ... )
    >>> func = RandomForestTSClassifierFunction(dataset="gunpoint")
    >>> result = func({"n_estimators": 100, "max_depth": 10})
    """

    name = "Random Forest Time-Series Classifier Function"
    _name_ = "random_forest_ts_classifier"
    __name__ = "RandomForestTSClassifierFunction"

    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5, 10]

    # Search space parameters
    para_names = ["n_estimators", "max_depth"]
    n_estimators_default = list(np.arange(10, 200, 10))
    max_depth_default = [None] + list(np.arange(2, 20))

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
            "max_depth": self.max_depth_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function with fixed dataset and cv."""
        X_raw, y = self._dataset_loader()
        X = extract_ts_features(X_raw)
        cv = self.cv

        def random_forest_ts_classifier(params: Dict[str, Any]) -> float:
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
                n_jobs=-1,
            )
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        self.pure_objective_function = random_forest_ts_classifier

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add fixed parameters for surrogate prediction."""
        return {
            **params,
            "dataset": self.dataset,
            "cv": self.cv,
        }
