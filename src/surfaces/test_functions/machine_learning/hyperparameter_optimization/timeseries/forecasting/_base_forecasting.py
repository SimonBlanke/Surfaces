# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .._base_timeseries import BaseTimeSeries


class BaseForecasting(BaseTimeSeries):
    """Base class for time-series forecasting test functions."""

    def _subsample_data(self, X, y, fidelity: float):
        """Sequential subsampling to preserve temporal order."""
        n_samples = max(1, int(len(X) * fidelity))
        return X[:n_samples], y[:n_samples]
