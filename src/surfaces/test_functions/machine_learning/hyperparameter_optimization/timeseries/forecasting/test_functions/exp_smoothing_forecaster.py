"""Exponential Smoothing Forecaster test function using sktime."""

from typing import Any, Dict, List, Optional

from surfaces._dependencies import check_dependency
from surfaces.modifiers import BaseModifier

from .._base_forecasting import BaseForecasting
from ..datasets import DATASETS


class ExpSmoothingForecasterFunction(BaseForecasting):
    """Exponential Smoothing (Holt-Winters) Forecaster test function.

    Uses sktime's ExponentialSmoothing for time-series forecasting with
    configurable trend and seasonal components.

    Parameters
    ----------
    dataset : str, default="airline"
        Dataset to use. One of: "airline", "energy", "sine_wave".
    forecast_horizon : int, default=12
        Number of steps to forecast ahead for evaluation.
    use_surrogate : bool, default=False
        If True, use pre-trained surrogate for fast evaluation.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds.

    Examples
    --------
    >>> from surfaces.test_functions.machine_learning.timeseries import (
    ...     ExpSmoothingForecasterFunction
    ... )
    >>> func = ExpSmoothingForecasterFunction(dataset="airline")
    >>> result = func({"trend": "add", "seasonal": "mul", "sp": 12})
    """

    name = "Exponential Smoothing Forecaster Function"
    _name_ = "exp_smoothing_forecaster"

    available_datasets = list(DATASETS.keys())

    # Search space parameters
    para_names = ["trend", "seasonal", "sp"]
    trend_default = [None, "add", "mul"]
    seasonal_default = [None, "add", "mul"]
    sp_default = [4, 6, 12, 24]

    def __init__(
        self,
        dataset: str = "airline",
        forecast_horizon: int = 12,
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

        self.dataset = dataset
        self.forecast_horizon = forecast_horizon
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
        """Search space containing hyperparameters."""
        return {
            "trend": self.trend_default,
            "seasonal": self.seasonal_default,
            "sp": self.sp_default,
        }

    def _ml_objective(self, params: Dict[str, Any]) -> float:
        import pandas as pd
        from sktime.forecasting.exp_smoothing import ExponentialSmoothing
        from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

        _, y_raw = self._dataset_loader()

        y = pd.Series(y_raw, index=pd.RangeIndex(start=0, stop=len(y_raw)))

        train_size = len(y) - self.forecast_horizon
        y_train = y[:train_size]
        y_test = y[train_size:]

        try:
            forecaster = ExponentialSmoothing(
                trend=params["trend"],
                seasonal=params["seasonal"],
                sp=params["sp"] if params["seasonal"] is not None else None,
                random_state=42,
            )

            forecaster.fit(y_train)
            fh = list(range(1, self.forecast_horizon + 1))
            y_pred = forecaster.predict(fh=fh)

            mape = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
            score = max(0.0, 1.0 - mape)
            return score

        except Exception:
            return 0.0

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add fixed parameters for surrogate prediction."""
        return {
            **params,
            "dataset": self.dataset,
            "forecast_horizon": self.forecast_horizon,
        }
