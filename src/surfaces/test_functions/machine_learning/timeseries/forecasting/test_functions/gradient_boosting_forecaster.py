"""Gradient Boosting Forecaster test function."""

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from .._base_forecasting import BaseForecasting
from ..datasets import DATASETS


def create_lagged_features(X, y, n_lags):
    """Create lagged features for time-series forecasting.

    Parameters
    ----------
    X : ndarray
        Time indices (not used directly, but kept for API consistency).
    y : ndarray
        Target time series values.
    n_lags : int
        Number of lag features to create.

    Returns
    -------
    X_lagged : ndarray
        Feature matrix with lagged values.
    y_lagged : ndarray
        Target values aligned with lagged features.
    """
    n_samples = len(y) - n_lags
    X_lagged = np.zeros((n_samples, n_lags))

    for i in range(n_lags):
        X_lagged[:, i] = y[n_lags - 1 - i : n_samples + n_lags - 1 - i]

    y_lagged = y[n_lags:]
    return X_lagged, y_lagged


class GradientBoostingForecasterFunction(BaseForecasting):
    """Gradient Boosting Forecaster test function.

    A time-series forecasting test function that uses lagged features
    with Gradient Boosting regression for prediction.

    Parameters
    ----------
    dataset : str, default="airline"
        Dataset to use. One of: "airline", "energy", "sine_wave".
    cv : int, default=5
        Number of time-series cross-validation splits.
    use_surrogate : bool, default=False
        If True, use pre-trained surrogate for fast evaluation.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds.

    Attributes
    ----------
    available_datasets : list
        Available dataset names.
    available_cv : list
        Available CV fold options.

    Examples
    --------
    >>> from surfaces.test_functions.machine_learning.timeseries import (
    ...     GradientBoostingForecasterFunction
    ... )
    >>> func = GradientBoostingForecasterFunction(dataset="airline")
    >>> func.search_space
    {'n_estimators': [...], 'max_depth': [...], 'n_lags': [...]}
    >>> result = func({"n_estimators": 50, "max_depth": 5, "n_lags": 12})
    """

    name = "Gradient Boosting Forecaster Function"
    _name_ = "gradient_boosting_forecaster"
    __name__ = "GradientBoostingForecasterFunction"

    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5]

    # Search space parameters
    para_names = ["n_estimators", "max_depth", "n_lags"]
    n_estimators_default = list(np.arange(10, 150, 10))
    max_depth_default = list(np.arange(2, 15))
    n_lags_default = list(np.arange(3, 25, 2))

    def __init__(
        self,
        dataset: str = "airline",
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
            "n_lags": self.n_lags_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function with fixed dataset and cv."""
        X, y = self._dataset_loader()
        cv = self.cv

        def gradient_boosting_forecaster(params: Dict[str, Any]) -> float:
            n_lags = params["n_lags"]

            # Create lagged features
            X_lagged, y_lagged = create_lagged_features(X, y, n_lags)

            model = GradientBoostingRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
            )

            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv)
            scores = []

            for train_idx, test_idx in tscv.split(X_lagged):
                X_train, X_test = X_lagged[train_idx], X_lagged[test_idx]
                y_train, y_test = y_lagged[train_idx], y_lagged[test_idx]

                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)  # R2 score
                scores.append(score)

            return np.mean(scores)

        self.pure_objective_function = gradient_boosting_forecaster

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add fixed parameters for surrogate prediction."""
        return {
            **params,
            "dataset": self.dataset,
            "cv": self.cv,
        }
