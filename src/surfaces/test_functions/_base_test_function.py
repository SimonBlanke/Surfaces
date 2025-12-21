# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class BaseTestFunction:
    """Base class for all test functions in the Surfaces library.

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.
    memory : bool, default=False
        If True, caches evaluated positions to avoid redundant computations.
        The cache key is the position as a tuple of sorted parameter values.

    Examples
    --------
    >>> func = SphereFunction(n_dim=2)
    >>> func({"x0": 1.0, "x1": 2.0})      # dict input
    >>> func(np.array([1.0, 2.0]))        # array input
    >>> func([1.0, 2.0])                  # list input
    """

    pure_objective_function: callable

    # =========================================================================
    # Spec: Function Characteristics (override in subclasses)
    # =========================================================================
    # All function metadata should be defined in _spec. This includes:
    # - name: Human-readable function name
    # - n_dim: Number of dimensions (None if variable)
    # - n_objectives: Number of objectives (1 for single-objective)
    # - default_bounds: Tuple of (min, max) for search space
    # - func_id: Function ID for benchmark suites (CEC, BBOB)
    # - Boolean flags: continuous, differentiable, convex, separable, unimodal, scalable

    _spec: Dict[str, Any] = {
        "name": None,
        "n_dim": None,
        "n_objectives": 1,
        "default_bounds": (-5.0, 5.0),
        "func_id": None,
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

    # Backward compatibility properties that read from spec
    @property
    def default_bounds(self) -> Tuple[float, float]:
        """Default parameter bounds for the search space."""
        return self.spec.get("default_bounds", (-5.0, 5.0))

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
    def __init__(self, objective="minimize", sleep=0, memory=False):
        if objective not in ("minimize", "maximize"):
            raise ValueError(f"objective must be 'minimize' or 'maximize', got '{objective}'")
        self.objective = objective
        self.sleep = sleep
        self.memory = memory
        self._memory_cache: Dict[Tuple, float] = {}

    def _create_objective_function(self):
        raise NotImplementedError("'_create_objective_function' must be implemented")

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space for this function (override in subclasses)."""
        raise NotImplementedError("'search_space' must be implemented")

    # =========================================================================
    # Primary Interface: __call__
    # =========================================================================

    def __call__(
        self, params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None, **kwargs
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

        if self.memory:
            cache_key = self._params_to_cache_key(params)
            if cache_key in self._memory_cache:
                return self._memory_cache[cache_key]
            result = self._evaluate(params)
            self._memory_cache[cache_key] = result
            return result

        return self._evaluate(params)

    def _normalize_input(
        self, params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Convert any input format to dict."""
        if isinstance(params, (np.ndarray, list, tuple)):
            param_names = sorted(self.search_space.keys())
            if len(params) != len(param_names):
                raise ValueError(f"Expected {len(param_names)} values, got {len(params)}")
            return {name: params[i] for i, name in enumerate(param_names)}

        if params is None:
            params = {}
        return {**params, **kwargs}

    def _params_to_cache_key(self, params: Dict[str, Any]) -> Tuple:
        """Convert params dict to a hashable cache key (sorted tuple of values)."""
        return tuple(params[k] for k in sorted(params.keys()))

    def _evaluate(self, params: Dict[str, Any]) -> float:
        """Evaluate with timing and objective transformation."""
        time.sleep(self.sleep)
        raw_value = self.pure_objective_function(params)

        if self.objective == "maximize":
            return -raw_value
        return raw_value
