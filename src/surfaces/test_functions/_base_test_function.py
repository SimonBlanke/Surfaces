# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np


class BaseTestFunction:
    """Base class for all test functions in the Surfaces library.

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Examples
    --------
    >>> func = SphereFunction(n_dim=2)
    >>> func({"x0": 1.0, "x1": 2.0})      # dict input
    >>> func(np.array([1.0, 2.0]))        # array input
    >>> func([1.0, 2.0])                  # list input
    """

    pure_objective_function: callable

    default_bounds: Tuple[float, float] = (-5.0, 5.0)

    # =========================================================================
    # Spec: Function Characteristics (override in subclasses)
    # =========================================================================

    _spec: Dict[str, Any] = {
        "continuous": True,
        "differentiable": True,
        "convex": False,
        "separable": False,
        "unimodal": False,
        "scalable": False,
    }

    @property
    def spec(self) -> Dict[str, Any]:
        """Function characteristics merged from class hierarchy (read-only)."""
        result = {}
        for klass in reversed(type(self).__mro__):
            if hasattr(klass, "_spec"):
                result.update(klass._spec)
        return result

    # =========================================================================
    # Global Optimum Information (override in subclasses)
    # =========================================================================

    f_global: Optional[float] = None
    x_global: Optional[np.ndarray] = None

    def _create_objective_function_(func):
        """Decorator that calls _create_objective_function after __init__."""
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self._create_objective_function()
        return wrapper

    @_create_objective_function_
    def __init__(self, objective="minimize", sleep=0):
        if objective not in ("minimize", "maximize"):
            raise ValueError(
                f"objective must be 'minimize' or 'maximize', got '{objective}'"
            )
        self.objective = objective
        self.sleep = sleep

    def _create_objective_function(self):
        raise NotImplementedError("'_create_objective_function' must be implemented")

    @property
    def default_search_space(self) -> Dict[str, Any]:
        """Default search space for this function (override in subclasses)."""
        raise NotImplementedError("'default_search_space' must be implemented")

    # =========================================================================
    # Primary Interface: __call__
    # =========================================================================

    def __call__(
        self, params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None,
        **kwargs
    ) -> float:
        """
        Evaluate the objective function.

        Args:
            params: Parameter values as dict, array, list, or tuple
            **kwargs: Parameters as keyword arguments (only with dict input)

        Returns:
            The objective function value

        Examples:
            func({"x0": 1.0, "x1": 2.0})     # dict
            func(np.array([1.0, 2.0]))       # array
            func([1.0, 2.0])                 # list
            func(x0=1.0, x1=2.0)             # kwargs
        """
        params = self._normalize_input(params, **kwargs)
        return self._evaluate(params)

    def _normalize_input(
        self, params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Convert any input format to dict."""
        if isinstance(params, (np.ndarray, list, tuple)):
            param_names = sorted(self.default_search_space.keys())
            if len(params) != len(param_names):
                raise ValueError(
                    f"Expected {len(param_names)} values, got {len(params)}"
                )
            return {name: params[i] for i, name in enumerate(param_names)}

        if params is None:
            params = {}
        return {**params, **kwargs}

    def _evaluate(self, params: Dict[str, Any]) -> float:
        """Evaluate with timing and objective transformation."""
        time.sleep(self.sleep)
        raw_value = self.pure_objective_function(params)

        if self.objective == "maximize":
            return -raw_value
        return raw_value
