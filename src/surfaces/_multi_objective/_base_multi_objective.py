# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for multi-objective optimization test functions."""

import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class MultiObjectiveFunction:
    """Base class for multi-objective test functions.

    Multi-objective functions return a vector of objective values instead of
    a scalar. The goal is typically to find the Pareto front - the set of
    solutions where no objective can be improved without worsening another.

    Parameters
    ----------
    n_dim : int
        Number of input dimensions.
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    n_objectives : int
        Number of objectives (set by subclass).
    n_dim : int
        Number of input dimensions.

    Examples
    --------
    >>> func = ZDT1(n_dim=30)
    >>> result = func(np.zeros(30))
    >>> result.shape
    (2,)
    """

    pure_objective_function: callable
    n_objectives: int = 2
    default_bounds: Tuple[float, float] = (0.0, 1.0)
    default_size: int = 1000

    _spec: Dict[str, Any] = {
        "continuous": True,
        "differentiable": True,
        "convex": False,
        "scalable": True,
    }

    @property
    def spec(self) -> Dict[str, Any]:
        """Function characteristics merged from class hierarchy (read-only)."""
        result = {}
        for klass in reversed(type(self).__mro__):
            if hasattr(klass, "_spec"):
                result.update(klass._spec)
        return result

    def _create_objective_function_(func):
        """Decorator that calls _create_objective_function after __init__."""

        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self._create_objective_function()

        return wrapper

    @_create_objective_function_
    def __init__(self, n_dim: int, sleep: float = 0):
        self.n_dim = n_dim
        self.sleep = sleep

    def _create_objective_function(self):
        raise NotImplementedError("'_create_objective_function' must be implemented")

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space based on default_bounds and n_dim."""
        min_val, max_val = self.default_bounds
        return self._create_search_space(min=min_val, max=max_val, size=self.default_size)

    def _create_search_space(
        self,
        min: float = 0.0,
        max: float = 1.0,
        size: int = 1000,
    ) -> Dict[str, Any]:
        """Create search space for the function."""
        search_space = {}
        dim_size = int(size ** (1 / self.n_dim))

        for dim in range(self.n_dim):
            dim_str = f"x{dim}"
            step_size = (max - min) / dim_size
            values = np.arange(min, max, step_size)
            search_space[dim_str] = values

        return search_space

    def __call__(
        self, params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None, **kwargs
    ) -> np.ndarray:
        """Evaluate the multi-objective function.

        Parameters
        ----------
        params : dict, array, list, or tuple
            Parameter values.

        Returns
        -------
        np.ndarray
            Vector of objective values with shape (n_objectives,).

        Examples
        --------
        >>> func({"x0": 0.5, "x1": 0.5})     # dict
        >>> func(np.array([0.5, 0.5]))       # array
        >>> func([0.5, 0.5])                 # list
        """
        params = self._normalize_input(params, **kwargs)
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

    def _evaluate(self, params: Dict[str, Any]) -> np.ndarray:
        """Evaluate with timing."""
        time.sleep(self.sleep)
        return self.pure_objective_function(params)

    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dict to numpy array."""
        return np.array([params[f"x{i}"] for i in range(self.n_dim)])

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        """Generate points on the theoretical Pareto front.

        Parameters
        ----------
        n_points : int, default=100
            Number of points to generate on the Pareto front.

        Returns
        -------
        np.ndarray
            Array of shape (n_points, n_objectives) containing points
            on the Pareto front.
        """
        raise NotImplementedError("'pareto_front' must be implemented by subclass")

    def pareto_set(self, n_points: int = 100) -> np.ndarray:
        """Generate points in the Pareto set (decision space).

        Parameters
        ----------
        n_points : int, default=100
            Number of points to generate in the Pareto set.

        Returns
        -------
        np.ndarray
            Array of shape (n_points, n_dim) containing points
            in the Pareto set.
        """
        raise NotImplementedError("'pareto_set' must be implemented by subclass")
