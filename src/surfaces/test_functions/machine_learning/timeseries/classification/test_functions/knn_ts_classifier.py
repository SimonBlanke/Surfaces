from typing import Any, Dict

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""K-Nearest Neighbors Time-Series Classifier using DTW-like distance."""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .._base_ts_classification import BaseTSClassification
from ..datasets import DATASETS


class KNNTSClassifierFunction(BaseTSClassification):
    """K-Nearest Neighbors Time-Series Classifier.

    Uses KNN on raw time-series data (flattened) with different distance
    metrics. A simple baseline approach for time-series classification.

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
    ...     KNNTSClassifierFunction
    ... )
    >>> func = KNNTSClassifierFunction(dataset="gunpoint")
    >>> result = func({"n_neighbors": 5, "metric": "euclidean"})
    """

    name = "KNN Time-Series Classifier Function"
    _name_ = "knn_ts_classifier"
    __name__ = "KNNTSClassifierFunction"

    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5, 10]

    # Search space parameters
    para_names = ["n_neighbors", "metric"]
    n_neighbors_default = list(np.arange(1, 30, 2))
    metric_default = ["euclidean", "manhattan", "chebyshev", "minkowski"]

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
            "n_neighbors": self.n_neighbors_default,
            "metric": self.metric_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function with fixed dataset and cv."""
        X_raw, y = self._dataset_loader()
        # Normalize time-series for better distance computation
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        cv = self.cv

        def knn_ts_classifier(params: Dict[str, Any]) -> float:
            model = KNeighborsClassifier(
                n_neighbors=params["n_neighbors"],
                metric=params["metric"],
                n_jobs=-1,
            )
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        self.pure_objective_function = knn_ts_classifier

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add fixed parameters for surrogate prediction."""
        return {
            **params,
            "dataset": self.dataset,
            "cv": self.cv,
        }
