# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
from typing import Dict, Any, Optional, Tuple

import numpy as np


class BaseTestFunction:
    """Base class for all test functions in the Surfaces library.

    This class provides the core interface for optimization test functions,
    including evaluation, search space definition, and integration with
    external optimization libraries.

    Primary interface:
        func(params)      - Evaluate the function (uses metric setting)
        func.loss(params) - Always returns value to minimize
        func.score(params) - Always returns value to maximize

    Parameters
    ----------
    metric : str, default="loss"
        Either "loss" (minimize) or "score" (maximize).
        Controls the return value of __call__().
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.
    validate : bool, default=True
        Whether to validate parameters against the search space.

    Examples
    --------
    >>> from surfaces.test_functions import SphereFunction
    >>> func = SphereFunction(n_dim=2)
    >>> result = func({"x0": 1.0, "x1": 2.0})
    """

    pure_objective_function: callable

    # Default bounds for the search space (override in subclasses)
    default_bounds: Tuple[float, float] = (-5.0, 5.0)

    def _create_objective_function_(func):
        """Decorator that calls _create_objective_function after __init__."""
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self._create_objective_function()
        return wrapper

    @_create_objective_function_
    def __init__(self, metric, sleep, validate=True):
        self.sleep = sleep
        self.metric = metric
        self._validate = validate

    def _create_objective_function(self):
        raise NotImplementedError("'_create_objective_function' must be implemented")

    @property
    def default_search_space(self) -> Dict[str, Any]:
        """Default search space for this function (override in subclasses)."""
        raise NotImplementedError("'default_search_space' must be implemented")

    def _return_metric(self, value):
        """Transform raw value based on metric setting (override in subclasses)."""
        return value

    # =========================================================================
    # Primary Interface: __call__
    # =========================================================================

    def __call__(
        self, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> float:
        """
        Evaluate the objective function.

        This is the primary interface for function evaluation. Supports
        multiple invocation styles for convenience.

        Args:
            params: Dictionary of parameter name -> value mappings
            **kwargs: Parameters as keyword arguments

        Returns:
            The objective function value

        Examples:
            # Dict style
            result = func({'x0': 1.0, 'x1': 2.0})

            # Kwargs style
            result = func(x0=1.0, x1=2.0)

            # Mixed style
            result = func({'x0': 1.0}, x1=2.0)
        """
        if params is None:
            params = {}
        params = {**params, **kwargs}

        if self._validate:
            self._validate_params(params)

        return self._evaluate_with_timing(params)

    def _evaluate_with_timing(self, params: Dict[str, Any]) -> float:
        """Evaluate with sleep timing applied."""
        time.sleep(self.sleep)
        raw_value = self.pure_objective_function(params)
        return self._return_metric(raw_value)

    def _evaluate_raw(self, params: Dict[str, Any]) -> float:
        """Evaluate without metric transformation (internal use)."""
        return self.pure_objective_function(params)

    # =========================================================================
    # Explicit loss/score Methods
    # =========================================================================

    def loss(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> float:
        """
        Evaluate and return a value to MINIMIZE.

        For functions that naturally return a loss (lower is better),
        this returns the raw value. For functions that return a score
        (higher is better), this returns the negated value.

        Args:
            params: Dictionary of parameter name -> value mappings
            **kwargs: Parameters as keyword arguments

        Returns:
            Loss value (lower is better)
        """
        if params is None:
            params = {}
        params = {**params, **kwargs}

        if self._validate:
            self._validate_params(params)

        time.sleep(self.sleep)
        raw_value = self.pure_objective_function(params)
        return self._to_loss(raw_value)

    def score(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> float:
        """
        Evaluate and return a value to MAXIMIZE.

        For functions that naturally return a score (higher is better),
        this returns the raw value. For functions that return a loss
        (lower is better), this returns the negated value.

        Args:
            params: Dictionary of parameter name -> value mappings
            **kwargs: Parameters as keyword arguments

        Returns:
            Score value (higher is better)
        """
        if params is None:
            params = {}
        params = {**params, **kwargs}

        if self._validate:
            self._validate_params(params)

        time.sleep(self.sleep)
        raw_value = self.pure_objective_function(params)
        return self._to_score(raw_value)

    def _to_loss(self, raw_value: float) -> float:
        """Convert raw value to loss (override in subclasses if needed)."""
        return raw_value

    def _to_score(self, raw_value: float) -> float:
        """Convert raw value to score (override in subclasses if needed)."""
        return -raw_value

    # =========================================================================
    # Input Validation
    # =========================================================================

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate that parameters match the expected search space.

        Validates that all required parameter keys are present and no
        unexpected keys are provided. Does not validate bounds since
        custom search spaces may use different ranges.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are missing or unexpected
        """
        space = self.default_search_space
        expected_keys = set(space.keys())
        provided_keys = set(params.keys())

        # Check for missing parameters
        missing = expected_keys - provided_keys
        if missing:
            raise ValueError(
                f"Missing required parameters: {missing}. "
                f"Expected: {expected_keys}"
            )

        # Check for unexpected parameters
        extra = provided_keys - expected_keys
        if extra:
            raise ValueError(
                f"Unexpected parameters: {extra}. "
                f"Expected: {expected_keys}"
            )

    # =========================================================================
    # Alternative Evaluation Methods
    # =========================================================================

    def evaluate(self, *args) -> float:
        """
        Evaluate using positional arguments.

        Arguments are mapped to parameters in sorted key order.

        Args:
            *args: Parameter values in order of sorted(search_space.keys())

        Returns:
            The objective function value

        Example:
            func.evaluate(1.0, 2.0, 3.0)  # For x0, x1, x2
        """
        param_names = sorted(self.search_space().keys())
        if len(args) != len(param_names):
            raise ValueError(
                f"Expected {len(param_names)} arguments for parameters "
                f"{param_names}, got {len(args)}"
            )
        params = {name: args[i] for i, name in enumerate(param_names)}
        return self(params)

    def evaluate_array(self, x: np.ndarray) -> float:
        """
        Evaluate using a numpy array.

        Array elements are mapped to parameters in sorted key order.
        This format is compatible with scipy.optimize.

        Args:
            x: 1D array of parameter values

        Returns:
            The objective function value

        Example:
            func.evaluate_array(np.array([1.0, 2.0, 3.0]))
        """
        return self.evaluate(*x)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate multiple points at once.

        Args:
            X: 2D array of shape (n_points, n_params)

        Returns:
            1D array of objective values

        Example:
            results = func.evaluate_batch(np.array([[1,2], [3,4], [5,6]]))
        """
        return np.array([self.evaluate_array(x) for x in X])

    # =========================================================================
    # scipy Integration
    # =========================================================================

    def to_scipy(self) -> Tuple[callable, "Bounds", np.ndarray]:
        """
        Convert to scipy.optimize compatible format.

        Returns:
            Tuple of (objective_function, bounds, x0) where:
            - objective_function: Callable taking numpy array
            - bounds: scipy.optimize.Bounds object
            - x0: Initial guess (center of bounds)

        Example:
            from scipy.optimize import minimize

            func = SphereFunction(n_dim=3)
            objective, bounds, x0 = func.to_scipy()

            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        """
        from scipy.optimize import Bounds

        space = self.search_space()
        param_names = sorted(space.keys())

        # Extract bounds
        lower = []
        upper = []
        for name in param_names:
            values = space[name]
            if hasattr(values, '__iter__') and not isinstance(values, str):
                values_list = list(values)
                numeric_values = [v for v in values_list if isinstance(v, (int, float))]
                if numeric_values:
                    lower.append(min(numeric_values))
                    upper.append(max(numeric_values))
                else:
                    raise ValueError(
                        f"Cannot create scipy bounds for non-numeric parameter '{name}'"
                    )
            else:
                raise ValueError(
                    f"Cannot create scipy bounds for parameter '{name}'"
                )

        lower = np.array(lower)
        upper = np.array(upper)
        x0 = (lower + upper) / 2

        # Create objective that uses loss (scipy minimizes)
        def objective(x: np.ndarray) -> float:
            params = {name: x[i] for i, name in enumerate(param_names)}
            # Bypass validation for performance in optimization loops
            time.sleep(self.sleep)
            raw_value = self.pure_objective_function(params)
            return self._to_loss(raw_value)

        return objective, Bounds(lower, upper), x0

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get parameter bounds as numpy arrays.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays

        Example:
            lower, upper = func.get_bounds()
        """
        space = self.search_space()
        param_names = sorted(space.keys())

        lower = []
        upper = []
        for name in param_names:
            values = space[name]
            if hasattr(values, '__iter__') and not isinstance(values, str):
                values_list = list(values)
                numeric_values = [v for v in values_list if isinstance(v, (int, float))]
                if numeric_values:
                    lower.append(min(numeric_values))
                    upper.append(max(numeric_values))
                else:
                    lower.append(float('-inf'))
                    upper.append(float('inf'))
            else:
                lower.append(float('-inf'))
                upper.append(float('inf'))

        return np.array(lower), np.array(upper)
