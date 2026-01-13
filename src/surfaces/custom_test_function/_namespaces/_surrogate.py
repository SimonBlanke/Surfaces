# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Surrogate namespace for CustomTestFunction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from .._custom_test_function import CustomTestFunction


class SurrogateNamespace:
    """Surrogate modeling tools for fast approximation.

    This namespace provides methods for fitting surrogate models
    to the collected evaluation data, enabling fast predictions
    without expensive function evaluations.

    Parameters
    ----------
    func : CustomTestFunction
        The parent function.

    Examples
    --------
    >>> func.surrogate.fit(method="gaussian_process")
    >>> prediction = func.surrogate.predict({"x": 0.5, "y": 0.5})
    >>> uncertainty = func.surrogate.uncertainty({"x": 0.5, "y": 0.5})
    """

    def __init__(self, func: "CustomTestFunction") -> None:
        self._func = func
        self._model = None
        self._method: Optional[str] = None
        self._is_fitted: bool = False

    def _check_data(self, min_evaluations: int = 10) -> None:
        """Check that sufficient data is available."""
        if self._func.n_evaluations < min_evaluations:
            raise ValueError(
                f"Surrogate fitting requires at least {min_evaluations} evaluations, "
                f"got {self._func.n_evaluations}"
            )

    def _check_fitted(self) -> None:
        """Check that a model has been fitted."""
        if not self._is_fitted:
            raise RuntimeError("No surrogate model fitted. Call .fit() first.")

    @property
    def is_fitted(self) -> bool:
        """Whether a surrogate model has been fitted."""
        return self._is_fitted

    @property
    def method(self) -> Optional[str]:
        """The method used for the current surrogate model."""
        return self._method

    def fit(
        self,
        method: str = "gaussian_process",
        **kwargs,
    ) -> "SurrogateNamespace":
        """Fit a surrogate model to the collected data.

        Parameters
        ----------
        method : str, default="gaussian_process"
            Surrogate method to use:
            - "gaussian_process": Gaussian Process (sklearn)
            - "random_forest": Random Forest (sklearn)
            - "gradient_boosting": Gradient Boosting (sklearn)
        **kwargs
            Additional arguments passed to the model constructor.

        Returns
        -------
        SurrogateNamespace
            Self, for method chaining.

        Examples
        --------
        >>> func.surrogate.fit(method="gaussian_process")
        >>> func.surrogate.fit(method="random_forest", n_estimators=100)
        """
        self._check_data()

        X, y = self._func.get_data_as_arrays()

        if method == "gaussian_process":
            self._model = self._fit_gaussian_process(X, y, **kwargs)
        elif method == "random_forest":
            self._model = self._fit_random_forest(X, y, **kwargs)
        elif method == "gradient_boosting":
            self._model = self._fit_gradient_boosting(X, y, **kwargs)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Use 'gaussian_process', 'random_forest', or 'gradient_boosting'."
            )

        self._method = method
        self._is_fitted = True
        return self

    def _fit_gaussian_process(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit a Gaussian Process model."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
        except ImportError:
            raise ImportError(
                "Gaussian Process requires scikit-learn. " "Install with: pip install scikit-learn"
            )

        kernel = kwargs.pop("kernel", Matern(nu=2.5))
        model = GaussianProcessRegressor(kernel=kernel, **kwargs)
        model.fit(X, y)
        return model

    def _fit_random_forest(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit a Random Forest model."""
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            raise ImportError(
                "Random Forest requires scikit-learn. " "Install with: pip install scikit-learn"
            )

        kwargs.setdefault("n_estimators", 100)
        kwargs.setdefault("random_state", 42)
        model = RandomForestRegressor(**kwargs)
        model.fit(X, y)
        return model

    def _fit_gradient_boosting(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit a Gradient Boosting model."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError:
            raise ImportError(
                "Gradient Boosting requires scikit-learn. " "Install with: pip install scikit-learn"
            )

        kwargs.setdefault("n_estimators", 100)
        kwargs.setdefault("random_state", 42)
        model = GradientBoostingRegressor(**kwargs)
        model.fit(X, y)
        return model

    def predict(
        self,
        params: Union[Dict[str, Any], np.ndarray, List[Dict[str, Any]]],
    ) -> Union[float, np.ndarray]:
        """Predict score(s) using the surrogate model.

        Parameters
        ----------
        params : dict, array, or list of dict
            Parameter values to predict. Can be:
            - Single dict: {"x": 0.5, "y": 0.5}
            - 2D array: shape (n_samples, n_dim)
            - List of dicts: [{"x": 0.5, "y": 0.5}, ...]

        Returns
        -------
        float or ndarray
            Predicted score(s).

        Examples
        --------
        >>> score = func.surrogate.predict({"x": 0.5, "y": 0.5})
        >>> scores = func.surrogate.predict(X_grid)
        """
        self._check_fitted()

        X = self._normalize_predict_input(params)
        predictions = self._model.predict(X)

        # Return scalar for single prediction
        if len(predictions) == 1:
            return float(predictions[0])
        return predictions

    def uncertainty(
        self,
        params: Union[Dict[str, Any], np.ndarray, List[Dict[str, Any]]],
    ) -> Union[float, np.ndarray]:
        """Get prediction uncertainty (standard deviation).

        Only available for Gaussian Process surrogate.

        Parameters
        ----------
        params : dict, array, or list of dict
            Parameter values.

        Returns
        -------
        float or ndarray
            Prediction standard deviation(s).

        Examples
        --------
        >>> std = func.surrogate.uncertainty({"x": 0.5, "y": 0.5})
        """
        self._check_fitted()

        if self._method != "gaussian_process":
            raise ValueError(
                f"Uncertainty is only available for Gaussian Process, " f"got {self._method}"
            )

        X = self._normalize_predict_input(params)
        _, std = self._model.predict(X, return_std=True)

        if len(std) == 1:
            return float(std[0])
        return std

    def _normalize_predict_input(
        self,
        params: Union[Dict[str, Any], np.ndarray, List[Dict[str, Any]]],
    ) -> np.ndarray:
        """Convert prediction input to 2D array."""
        param_names = self._func.param_names

        if isinstance(params, np.ndarray):
            if params.ndim == 1:
                return params.reshape(1, -1)
            return params

        if isinstance(params, dict):
            return np.array([[params[name] for name in param_names]])

        if isinstance(params, list) and isinstance(params[0], dict):
            return np.array([[p[name] for name in param_names] for p in params])

        raise ValueError(f"Invalid input type: {type(params)}")

    def suggest_next(self, n_suggestions: int = 1) -> List[Dict[str, float]]:
        """Suggest next point(s) to evaluate using acquisition function.

        Uses Expected Improvement for Gaussian Process, or
        random sampling from low-prediction regions for other methods.

        Parameters
        ----------
        n_suggestions : int, default=1
            Number of suggestions to return.

        Returns
        -------
        list of dict
            Suggested parameter configurations.

        Examples
        --------
        >>> suggestions = func.surrogate.suggest_next(n_suggestions=5)
        >>> for params in suggestions:
        ...     score = func(params)
        """
        self._check_fitted()

        # Generate candidate points
        n_candidates = 1000
        candidates = self._generate_candidates(n_candidates)

        # Score candidates
        predictions = self._model.predict(candidates)

        if self._method == "gaussian_process":
            # Use Expected Improvement
            _, stds = self._model.predict(candidates, return_std=True)
            best_so_far = self._func.best_score

            # EI calculation (for minimization)
            from scipy.stats import norm

            z = (best_so_far - predictions) / (stds + 1e-10)
            ei = (best_so_far - predictions) * norm.cdf(z) + stds * norm.pdf(z)
            scores = ei
        else:
            # Use prediction (lower is better for minimization)
            if self._func.objective == "minimize":
                scores = -predictions
            else:
                scores = predictions

        # Select top candidates
        top_indices = np.argsort(scores)[-n_suggestions:]

        # Convert to dicts
        param_names = self._func.param_names
        suggestions = []
        for idx in top_indices:
            suggestion = {name: float(candidates[idx, i]) for i, name in enumerate(param_names)}
            suggestions.append(suggestion)

        return suggestions

    def _generate_candidates(self, n_candidates: int) -> np.ndarray:
        """Generate random candidate points within bounds."""
        bounds = self._func.bounds
        param_names = self._func.param_names

        candidates = np.zeros((n_candidates, len(param_names)))
        for i, name in enumerate(param_names):
            low, high = bounds[name]
            candidates[:, i] = np.random.uniform(low, high, n_candidates)

        return candidates

    def score(self) -> float:
        """Get R^2 score of the surrogate model on training data.

        Returns
        -------
        float
            R^2 score (1.0 is perfect fit).
        """
        self._check_fitted()

        X, y = self._func.get_data_as_arrays()
        return float(self._model.score(X, y))
