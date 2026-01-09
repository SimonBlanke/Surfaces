# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from surfaces._array_utils import ArrayLike, is_array_like
from surfaces.modifiers import BaseModifier

from ._mixins import (
    CallbackMixin,
    DataCollectionMixin,
    ModifierMixin,
    VisualizationMixin,
)


class BaseTestFunction(
    CallbackMixin,
    DataCollectionMixin,
    ModifierMixin,
    VisualizationMixin,
):
    """Base class for all test functions in the Surfaces library.

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations. Modifiers are
        applied in the order they appear in the list.
    memory : bool, default=False
        If True, caches evaluated positions to avoid redundant computations.
    collect_data : bool, default=True
        If True, collects evaluation data including search_data, best_score,
        best_params, n_evaluations, and total_time.
    callbacks : callable or list of callables, optional
        Function(s) called after each evaluation with the record dict.
    catch_errors : dict, optional
        Dictionary mapping exception types to return values. Use ... (Ellipsis)
        as a catch-all key for any unmatched exceptions.

    Attributes
    ----------
    n_evaluations : int
        Number of function evaluations performed.
    search_data : list of dict
        History of all evaluations as list of dicts containing parameters and score.
    best_score : float or None
        Best score found (respects objective direction).
    best_params : dict or None
        Parameters that achieved the best score.
    total_time : float
        Cumulative time spent in function evaluations (seconds).

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

    # Type alias for callbacks
    CallbackType = Union[Callable[[Dict[str, Any]], None], List[Callable[[Dict[str, Any]], None]]]

    @_create_objective_function_
    def __init__(
        self,
        objective="minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
    ):
        if objective not in ("minimize", "maximize"):
            raise ValueError(f"objective must be 'minimize' or 'maximize', got '{objective}'")
        self.objective = objective
        self.memory = memory
        self.collect_data = collect_data
        self.catch_errors: Optional[Dict[Type[Exception], float]] = catch_errors
        self._memory_cache: Dict[Tuple, float] = {}

        # Initialize mixins
        self._init_callbacks(callbacks)
        self._init_data_collection()
        self._init_modifiers(modifiers)

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
        self,
        params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None,
        **kwargs,
    ) -> float:
        """Evaluate the objective function.

        Args:
            params: Parameter values as dict, array, list, or tuple
            **kwargs: Parameters as keyword arguments (only with dict input)

        Returns:
            The objective function value
        """
        params = self._normalize_input(params, **kwargs)

        if self.memory:
            cache_key = self._params_to_cache_key(params)
            if cache_key in self._memory_cache:
                result = self._memory_cache[cache_key]
                if self.collect_data or self._callbacks:
                    self._record_evaluation(params, result, from_cache=True)
                return result

        start_time = time.perf_counter()
        result = self._evaluate(params)
        elapsed_time = time.perf_counter() - start_time

        if self.memory:
            cache_key = self._params_to_cache_key(params)
            self._memory_cache[cache_key] = result

        if self.collect_data or self._callbacks:
            self._record_evaluation(params, result, elapsed_time=elapsed_time)

        return result

    def _normalize_input(
        self,
        params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None,
        **kwargs,
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
        """Evaluate with modifiers and objective transformation."""
        try:
            raw_value = self.pure_objective_function(params)
        except Exception as e:
            if self.catch_errors is not None:
                for exc_type, return_value in self.catch_errors.items():
                    if exc_type is ... or isinstance(e, exc_type):
                        return return_value
            raise

        # Apply modifiers if configured
        if self._modifiers:
            context = {
                "evaluation_count": self.n_evaluations,
                "best_score": self.best_score,
                "search_data": self.search_data,
            }
            for modifier in self._modifiers:
                raw_value = modifier.apply(raw_value, params, context)

        if self.objective == "maximize":
            return -raw_value
        return raw_value

    # =========================================================================
    # Reset Methods
    # =========================================================================

    def reset_memory(self) -> None:
        """Clear the memory cache."""
        self._memory_cache = {}

    def reset(self) -> None:
        """Reset all state including collected data and memory cache."""
        self.reset_data()
        self.reset_memory()

    # =========================================================================
    # Batch Evaluation
    # =========================================================================

    def batch(self, X: ArrayLike) -> ArrayLike:
        """Evaluate multiple parameter sets in a single call.

        Parameters
        ----------
        X : ArrayLike
            2D array of shape (n_points, n_dim) where each row is a parameter set.

        Returns
        -------
        ArrayLike
            1D array of shape (n_points,) with evaluation results.

        Raises
        ------
        NotImplementedError
            If the function does not implement _batch_objective.
        ValueError
            If X has wrong number of dimensions or wrong n_dim.
        """
        if not hasattr(self, "_batch_objective"):
            raise NotImplementedError(
                f"{type(self).__name__} does not support vectorized batch evaluation. "
                "Implement _batch_objective(X) to enable this feature."
            )

        if not is_array_like(X):
            raise TypeError(
                f"Expected array-like input with shape (n_points, n_dim), "
                f"got {type(X).__name__}"
            )

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array with shape (n_points, n_dim), got {X.ndim}D array")

        if X.shape[1] != self.n_dim:
            raise ValueError(f"Expected {self.n_dim} dimensions, got {X.shape[1]}")

        result = self._batch_objective(X)

        if self.objective == "maximize":
            result = -result

        return result
