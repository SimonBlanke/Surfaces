"""Exponential Smoothing Forecaster test function using sktime."""

from typing import Any, Dict

from .._base_forecasting import BaseForecasting
from ..datasets import DATASETS


def _check_sktime():
    """Check if sktime is available."""
    try:
        import sktime  # noqa: F401

        return True
    except ImportError:
        raise ImportError(
            "Exponential Smoothing forecaster requires sktime. "
            "Install with: pip install surfaces[timeseries]"
        )


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
    __name__ = "ExpSmoothingForecasterFunction"

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

        self.dataset = dataset
        self.forecast_horizon = forecast_horizon
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
            "trend": self.trend_default,
            "seasonal": self.seasonal_default,
            "sp": self.sp_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function with fixed dataset."""
        import pandas as pd
        from sktime.forecasting.exp_smoothing import ExponentialSmoothing
        from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

        _, y_raw = self._dataset_loader()
        forecast_horizon = self.forecast_horizon

        # Convert to pandas Series with proper index for sktime
        y = pd.Series(y_raw, index=pd.RangeIndex(start=0, stop=len(y_raw)))

        # Train/test split
        train_size = len(y) - forecast_horizon
        y_train = y[:train_size]
        y_test = y[train_size:]

        def exp_smoothing_forecaster(params: Dict[str, Any]) -> float:
            try:
                forecaster = ExponentialSmoothing(
                    trend=params["trend"],
                    seasonal=params["seasonal"],
                    sp=params["sp"] if params["seasonal"] is not None else None,
                    random_state=42,
                )

                forecaster.fit(y_train)
                fh = list(range(1, forecast_horizon + 1))
                y_pred = forecaster.predict(fh=fh)

                # Calculate MAPE and convert to score (1 - MAPE)
                mape = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
                # Clamp score to [0, 1] range
                score = max(0.0, 1.0 - mape)
                return score

            except Exception:
                # Invalid parameter combinations return low score
                return 0.0

        self.pure_objective_function = exp_smoothing_forecaster

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add fixed parameters for surrogate prediction."""
        return {
            **params,
            "dataset": self.dataset,
            "forecast_horizon": self.forecast_horizon,
        }
