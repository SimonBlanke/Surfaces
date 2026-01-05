# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from surfaces.modifiers import BaseModifier, ModifierPipeline


class BaseTestFunction:
    """Base class for all test functions in the Surfaces library.

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations. Modifiers are
        applied in the order they appear in the list. Examples include
        noise (GaussianNoise, UniformNoise, MultiplicativeNoise) and
        delays (DelayModifier). See surfaces.modifiers and surfaces.noise
        modules for available modifiers.
    memory : bool, default=False
        If True, caches evaluated positions to avoid redundant computations.
        The cache key is the position as a tuple of sorted parameter values.
    collect_data : bool, default=True
        If True, collects evaluation data including search_data, best_score,
        best_params, n_evaluations, and total_time. Set to False to disable
        tracking for performance-critical applications.
    callbacks : callable or list of callables, optional
        Function(s) called after each evaluation with the record dict.
        Signature: callback(record: dict) -> None
        The record contains all parameters plus 'score'.
    catch_errors : dict, optional
        Dictionary mapping exception types to return values. When an
        exception of a specified type occurs during evaluation, the
        corresponding value is returned instead of propagating the error.
        Use ... (Ellipsis) as a catch-all key for any unmatched exceptions.
        Exceptions not matching any key will still propagate normally.

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
    Basic usage with different input formats:

    >>> func = SphereFunction(n_dim=2)
    >>> func({"x0": 1.0, "x1": 2.0})      # dict input
    >>> func(np.array([1.0, 2.0]))        # array input
    >>> func([1.0, 2.0])                  # list input
    >>> func.n_evaluations                 # 3
    >>> func.best_score                    # best value seen
    >>> func.search_data                   # [{"x0": 1.0, "x1": 2.0, "score": 5.0}, ...]

    Callbacks Example
    -----------------
    Callbacks are invoked after each evaluation with a record dict containing
    all parameters and the score. Use callbacks for logging, streaming to
    external systems, or custom processing.

    Single callback:

    >>> records = []
    >>> func = SphereFunction(n_dim=2, callbacks=lambda r: records.append(r))
    >>> func([1.0, 2.0])
    >>> print(records)  # [{"x0": 1.0, "x1": 2.0, "score": 5.0}]

    Multiple callbacks:

    >>> func = SphereFunction(
    ...     n_dim=2,
    ...     callbacks=[
    ...         lambda r: print(f"Score: {r['score']}"),
    ...         lambda r: my_database.insert(r),
    ...     ]
    ... )

    Adding callbacks at runtime:

    >>> func = SphereFunction(n_dim=2)
    >>> func.add_callback(lambda r: print(r))
    >>> func([1.0, 2.0])  # prints the record
    >>> func.clear_callbacks()

    Catch Errors Example
    --------------------
    Use catch_errors to handle exceptions during evaluation gracefully.
    This is useful for optimization where some parameter combinations
    may cause numerical errors (division by zero, log of negative, etc.).
    The optimizer can continue exploring while the return value guides
    it away from problematic regions.

    Catch specific exceptions with custom return values:

    >>> func = SphereFunction(
    ...     n_dim=2,
    ...     catch_errors={
    ...         ZeroDivisionError: float('inf'),
    ...         ValueError: 1000.0,
    ...     }
    ... )
    >>> # ZeroDivisionError returns inf
    >>> # ValueError returns 1000.0
    >>> # Other exceptions still propagate

    Use ... (Ellipsis) as a catch-all for any unmatched exceptions:

    >>> func = SphereFunction(
    ...     n_dim=2,
    ...     catch_errors={
    ...         ValueError: 1000.0,   # Specific handling
    ...         ...: float('inf'),    # Everything else
    ...     }
    ... )
    >>> # ValueError returns 1000.0
    >>> # Any other exception returns inf

    Simple catch-all pattern:

    >>> func = SphereFunction(
    ...     n_dim=2,
    ...     catch_errors={...: float('inf')}
    ... )

    Modifiers Example
    -----------------
    Add noise to simulate measurement uncertainty:

    >>> from surfaces.modifiers import GaussianNoise
    >>> func = SphereFunction(
    ...     n_dim=2,
    ...     modifiers=[GaussianNoise(sigma=0.1, seed=42)]
    ... )
    >>> result = func([1.0, 2.0])  # Returns noisy evaluation
    >>> true_result = func.true_value([1.0, 2.0])  # Without modifiers

    Combine multiple modifiers (applied in order):

    >>> from surfaces.modifiers import DelayModifier
    >>> from surfaces.modifiers import GaussianNoise
    >>> func = SphereFunction(
    ...     n_dim=2,
    ...     modifiers=[
    ...         DelayModifier(delay=0.01),  # Applied first
    ...         GaussianNoise(sigma=0.1)     # Applied second
    ...     ]
    ... )

    Decaying noise over optimization:

    >>> func = SphereFunction(
    ...     n_dim=2,
    ...     modifiers=[
    ...         GaussianNoise(
    ...             sigma=0.5,
    ...             sigma_final=0.01,
    ...             schedule="linear",
    ...             total_evaluations=1000
    ...         )
    ...     ]
    ... )
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
        self._modifiers: ModifierPipeline = ModifierPipeline(modifiers if modifiers is not None else [])
        self._memory_cache: Dict[Tuple, float] = {}

        # Normalize callbacks to list
        if callbacks is None:
            self._callbacks: List[Callable] = []
        elif callable(callbacks):
            self._callbacks = [callbacks]
        else:
            self._callbacks = list(callbacks)

        # Data collection attributes
        self.n_evaluations: int = 0
        self.search_data: list = []
        self.best_score: Optional[float] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.total_time: float = 0.0

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
                result = self._memory_cache[cache_key]
                if self.collect_data or self._callbacks:
                    self._record_evaluation(params, result, from_cache=True)
                return result

        start_time = time.time()
        result = self._evaluate(params)
        elapsed_time = time.time() - start_time

        if self.memory:
            cache_key = self._params_to_cache_key(params)
            self._memory_cache[cache_key] = result

        if self.collect_data or self._callbacks:
            self._record_evaluation(params, result, elapsed_time=elapsed_time)

        return result

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
        """Evaluate with modifiers and objective transformation.

        If catch_errors is provided, exceptions matching the specified types
        return the corresponding value instead of propagating. Use ... (Ellipsis)
        as a catch-all key for any unmatched exceptions.

        Modifiers are applied after error handling but before the objective transform.
        """
        try:
            raw_value = self.pure_objective_function(params)
        except Exception as e:
            if self.catch_errors is not None:
                # Check if this exception type should be caught
                for exc_type, return_value in self.catch_errors.items():
                    if exc_type is ... or isinstance(e, exc_type):
                        return return_value
            raise

        # Apply modifiers if configured
        if len(self._modifiers) > 0:
            context = {
                "evaluation_count": self.n_evaluations,
                "best_score": self.best_score,
                "search_data": self.search_data,
            }
            raw_value = self._modifiers.apply(raw_value, params, context)

        if self.objective == "maximize":
            return -raw_value
        return raw_value

    def _record_evaluation(
        self,
        params: Dict[str, Any],
        score: float,
        elapsed_time: float = 0.0,
        from_cache: bool = False,
    ) -> None:
        """Record an evaluation and invoke callbacks."""
        record = {**params, "score": score}

        if self.collect_data:
            self.n_evaluations += 1
            self.search_data.append(record)

            # Update timing (only for non-cached evaluations)
            if not from_cache:
                self.total_time += elapsed_time

            # Update best score/params
            is_better = (
                self.best_score is None
                or (self.objective == "minimize" and score < self.best_score)
                or (self.objective == "maximize" and score > self.best_score)
            )
            if is_better:
                self.best_score = score
                self.best_params = params.copy()

        # Invoke callbacks
        for callback in self._callbacks:
            callback(record)

    def reset_data(self) -> None:
        """Reset all collected evaluation data.

        Clears n_evaluations, search_data, best_score, best_params, and total_time.
        Does not clear the memory cache (use reset_memory() for that).
        """
        self.n_evaluations = 0
        self.search_data = []
        self.best_score = None
        self.best_params = None
        self.total_time = 0.0

    def reset_memory(self) -> None:
        """Clear the memory cache."""
        self._memory_cache = {}

    def reset(self) -> None:
        """Reset all state including collected data and memory cache."""
        self.reset_data()
        self.reset_memory()

    # =========================================================================
    # Modifier Management
    # =========================================================================

    def true_value(
        self, params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None, **kwargs
    ) -> float:
        """Evaluate the function without modifiers.

        Returns the true (deterministic) function value, bypassing any
        configured modifiers. Useful for analysis and comparison.

        This method does not update search_data, n_evaluations, or callbacks.
        It also ignores memory caching.

        Parameters
        ----------
        params : dict, array, list, or tuple
            Parameter values to evaluate.
        **kwargs : dict
            Parameters as keyword arguments.

        Returns
        -------
        float
            The true function value without modifiers.

        Examples
        --------
        >>> from surfaces.modifiers import GaussianNoise
        >>> func = SphereFunction(
        ...     n_dim=2,
        ...     modifiers=[GaussianNoise(sigma=0.1, seed=42)]
        ... )
        >>> modified = func([1.0, 2.0])
        >>> true = func.true_value([1.0, 2.0])
        >>> print(f"Difference: {modified - true:.4f}")
        """
        params = self._normalize_input(params, **kwargs)
        raw_value = self.pure_objective_function(params)
        if self.objective == "maximize":
            return -raw_value
        return raw_value

    def reset_modifiers(self) -> None:
        """Reset all modifiers' internal state.

        Resets evaluation counters, random states, and any other
        stateful components in the modifier pipeline.
        """
        self._modifiers.reset()

    @property
    def modifiers(self) -> ModifierPipeline:
        """The modifier pipeline for this function."""
        return self._modifiers

    # =========================================================================
    # Callback Management
    # =========================================================================

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback to be invoked after each evaluation.

        Parameters
        ----------
        callback : callable
            Function that takes a record dict with parameters and 'score'.

        Examples
        --------
        >>> func = SphereFunction(n_dim=2)
        >>> func.add_callback(lambda r: print(f"Score: {r['score']}"))
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a previously added callback.

        Parameters
        ----------
        callback : callable
            The callback to remove.

        Raises
        ------
        ValueError
            If the callback is not in the list.
        """
        self._callbacks.remove(callback)

    def clear_callbacks(self) -> None:
        """Remove all callbacks."""
        self._callbacks = []

    @property
    def callbacks(self) -> List[Callable[[Dict[str, Any]], None]]:
        """List of registered callbacks (read-only copy)."""
        return self._callbacks.copy()
