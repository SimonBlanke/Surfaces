import numpy as np
from typing import Any, Dict, List, Optional
from numpy.lib.stride_tricks import sliding_window_view

# Surfaces library base class and data
from .._base_forecasting import BaseForecasting
from ..datasets import DATASETS


def apply_time_series_features(
    y: np.ndarray,
    n_lags: int,
    rolling_window: int,
    differencing: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised learning features from a univariate time series.

    Parameters
    ----------
    y            : 1-D array of observations (oldest → newest)
    n_lags       : number of lag features (0 = none)
    rolling_window: window size for rolling mean/std (0 = skip)
    differencing : order of differencing applied before feature extraction
    """
    if n_lags == 0 and rolling_window == 0:
        raise ValueError("At least one of n_lags or rolling_window must be > 0.")

    if differencing > 0:
        y = np.diff(y, n=differencing)

    n_samples = len(y)
    offset = max(n_lags, rolling_window)

    if n_samples <= offset:
        raise ValueError(
            f"Series length {n_samples} is too short for "
            f"n_lags={n_lags} / rolling_window={rolling_window}."
        )

    features = []

    # Lag features
    for lag in range(1, n_lags + 1):
        features.append(y[offset - lag : n_samples - lag])

    # Vectorised rolling statistics
    if rolling_window > 0:
        windows = sliding_window_view(y, window_shape=rolling_window)
        # windows has shape (n_samples - rolling_window + 1, rolling_window)
        # align to the same offset used by lag features
        start = offset - rolling_window
        features.append(windows[start:].mean(axis=1))
        features.append(windows[start:].std(axis=1, ddof=1))

    X = np.column_stack(features)
    y_target = y[offset:]

    return X, y_target

class TimeSeriesPipelineForecasterFunction(BaseForecasting):
    """
    A hyperparameter-searchable time series forecasting pipeline that combines:
      - Lag features and rolling statistics for feature engineering
      - Optional differencing for stationarity
      - Choice of scaler (none / standard / minmax)
      - Choice of model (Ridge / RandomForest / GradientBoosting)
      - Model-specific regularization parameters

    The objective function returns negative MAE (higher = better),
    compatible with a maximising optimiser.

    Parameters
    ----------
    dataset      : Name of the dataset to load (must be a key in DATASETS).
    objective    : Optimisation direction, default "maximize".
    modifiers    : Optional list of BaseModifier instances.
    memory       : Whether to enable caching in the base class.
    collect_data : Whether to collect evaluation data in the base class.
    train_size   : Fraction of data used for training (default 0.8).
    **kwargs     : Passed through to BaseForecasting.
    """

    _name_ = "time_series_pipeline_forecaster"
    _dependencies = {"ml": ["sklearn"]}

    para_names = [
        "n_lags",
        "rolling_window",
        "differencing",
        "scaler",
        "model",
        "model__regularization"
    ]

    n_lags_default = [3, 5, 7, 10, 14, 21]
    rolling_window_default = [0, 3, 7, 14]
    differencing_default = [0, 1, 2]
    scaler_default = ["none", "standard", "minmax"]
    model_default = ["ridge", "rf", "gb"]
    model__regularization_default = [0.001, 0.01, 0.1, 1.0, 10.0]

    def _default_search_space(self) -> Dict[str, List]:
        """Define the default hyperparameter search space for this function."""

        return {
            "n_lags": [3, 5, 7, 10, 14, 21],
            "rolling_window": [0, 3, 7, 14],
            "differencing": [0, 1, 2],
            "scaler": ["none", "standard", "minmax"],
            "model": ["ridge", "rf", "gb"],
            "model__regularization": [0.001, 0.01, 0.1, 1.0, 10.0],
        }
    
    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Include fixed parameters for surrogate model support."""
        return {
            **params,
            "dataset": self.dataset,
            "train_size": self.train_size
        }
    
    def __init__(
        self,
        dataset: str = "airline",
        objective: str = "maximize",
        modifiers: Optional[List] = None,
        memory: bool = False,
        collect_data: bool = True,
        train_size: float = 0.8,
        **kwargs: Any,
    ) -> None:
        if dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available datasets: {list(DATASETS.keys())}"
            )
        if not 0.0 < train_size < 1.0:
            raise ValueError(
                f"train_size must be between 0 and 1 exclusive, got {train_size}."
            )

        self.dataset = dataset
        self.train_size = train_size
        self._dataset_loader = DATASETS[dataset]
        self._cached_data: Optional[tuple] = None

        super().__init__(
            objective=objective,
            modifiers=modifiers,
            memory=memory,
            collect_data=collect_data,
            **kwargs,
        )


    # Data loading


    def _get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load and cache the dataset.  Returns (X_raw, y_raw) where
        y_raw is the univariate target series used for feature engineering.
        """
        if self._cached_data is None:
            self._cached_data = self._dataset_loader()
        return self._cached_data


    # Scaler factory


    @staticmethod
    def _build_scaler(scaler_type: str):
        """Return a fitted-ready scaler instance, or None for 'none'."""
        if scaler_type == "standard":
            return StandardScaler()
        if scaler_type == "minmax":
            return MinMaxScaler()
        if scaler_type == "none":
            return None
        raise ValueError(f"Unknown scaler type: {scaler_type!r}")


    # Model factory


    @staticmethod
    def _build_model(model_type: str, reg: float):
        """
        Construct a scikit-learn regressor from the model type and the
        shared regularization parameter, mapped per-model as follows:

            ridge -> alpha        (float, e.g. 0.001 – 10.0)
            rf    -> max_depth    (int cast of reg, clipped to >= 1)
            gb    -> learning_rate (float, e.g. 0.001 – 1.0)
        """
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

        if model_type == "ridge":
            return Ridge(alpha=reg)

        if model_type == "rf":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=max(1, int(reg)),
                random_state=42,
            )

        if model_type == "gb":
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=float(np.clip(reg, 1e-4, 1.0)),
                random_state=42,
            )

        raise ValueError(f"Unknown model type: {model_type!r}")


    # Objective


    def _ml_objective(self, params: Dict[str, Any]) -> float:
        """
        Evaluate a single hyperparameter configuration.

        Steps
        -----
        1. Load (cached) raw series.
        2. Apply differencing, lag features, and rolling statistics.
        3. Chronological train/test split.
        4. Optionally scale features.
        5. Fit the chosen model and return negative MAE.

        Returns
        -------
        float
            Negative MAE — higher is better, compatible with maximisation.
        """
        # model and preprocessing
        from sklearn.metrics import mean_absolute_error

        # 1. Raw data
        _, y_raw = self._get_training_data()

        # 2. Feature engineering
        try:
            X, y = apply_time_series_features(
                y_raw,
                n_lags=params["n_lags"],
                rolling_window=params["rolling_window"],
                differencing=params["differencing"],
            )
        except ValueError as exc:
            # Config produced an unusable feature matrix (e.g. series too short)
            # Return a very poor score so the optimiser discards this config.
            return -float("inf")

        # 3. Chronological split
        split_idx = int(len(X) * self.train_size)
        if split_idx == 0 or split_idx == len(X):
            # Degenerate split — not enough data for this param combination
            return -float("inf")

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 4. Scaling
        scaler = self._build_scaler(params["scaler"])
        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # 5. Model training and evaluation
        model = self._build_model(params["model"], params["model__regularization"])
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))

        return -mae


    # Dunder helpers


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset={self.dataset!r}, "
            f"train_size={self.train_size!r})"
        )