# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from surfaces._array_utils import ArrayLike, is_array_like
from surfaces.modifiers import BaseModifier


class BaseTestFunction:
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

    Examples
    --------
    >>> func = SphereFunction(n_dim=2)
    >>> func({"x0": 1.0, "x1": 2.0})      # dict input
    >>> func(np.array([1.0, 2.0]))        # array input
    >>> func([1.0, 2.0])                  # list input
    """

    @property
    def __name__(self):
        """Make __name__ accessible on instances (external libs expect it)."""
        return type(self).__name__

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-derive name if not explicitly defined
        if "name" not in cls.__dict__:
            spec = cls.__dict__.get("_spec", {})
            if isinstance(spec, dict) and "name" in spec:
                cls.name = spec["name"]
            else:
                raw = cls.__name__.removesuffix("Function")
                cls.name = (
                    re.sub(
                        r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
                        " ",
                        raw,
                    )
                    + " Function"
                )
        # Auto-derive _name_ if not explicitly defined
        if "_name_" not in cls.__dict__:
            cls._name_ = cls.name.lower().replace(" ", "_")

    # =========================================================================
    # Spec: Function Characteristics (override in subclasses)
    # =========================================================================

    _spec: Dict[str, Any] = {
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

    # =========================================================================
    # Global Optimum Information (override in subclasses)
    # =========================================================================

    f_global: Optional[float] = None
    x_global: Optional[np.ndarray] = None

    # Optional dependencies: {extras_group: [package, ...]}
    # Subclasses set this to declare packages checked at init time.
    _dependencies = None

    # Type alias for callbacks
    CallbackType = Union[Callable[[Dict[str, Any]], None], List[Callable[[Dict[str, Any]], None]]]

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
        self.collect_data = collect_data

        # Private state: memory
        self._memory_enabled: bool = memory
        self._memory_cache: Dict[Tuple, float] = {}

        # Private state: callbacks
        if callbacks is None:
            self._callbacks: List[Callable] = []
        elif callable(callbacks):
            self._callbacks = [callbacks]
        else:
            self._callbacks = list(callbacks)

        # Private state: data collection
        self._n_evaluations: int = 0
        self._search_data: List[Dict[str, Any]] = []
        self._best_score: Optional[float] = None
        self._best_params: Optional[Dict[str, Any]] = None
        self._total_time: float = 0.0

        # Private state: modifiers
        self._modifiers: List[BaseModifier] = modifiers if modifiers is not None else []

        # Private state: error handlers
        self._error_handlers: Optional[Dict[Type[Exception], float]] = catch_errors

        self._check_dependencies()

        # Accessor caches (lazy-loaded)
        self._spec_accessor = None
        self._data_accessor = None
        self._callbacks_accessor = None
        self._modifiers_accessor = None
        self._memory_accessor = None
        self._errors_accessor = None
        self._meta_accessor = None

    def _check_dependencies(self):
        """Check that required optional dependencies are installed.

        Iterates over ``_dependencies`` (if set) and calls
        :func:`surfaces._dependencies.check_dependency` for each package.
        Subclasses may override to customise behaviour (e.g. skip when
        using a surrogate model).
        """
        if self._dependencies:
            from surfaces._dependencies import check_dependency

            for extras, packages in self._dependencies.items():
                for package in packages:
                    check_dependency(package, extras)

    def _objective(self, params: Dict[str, Any]) -> float:
        """Compute the raw objective value for the given parameters.

        Override this method in subclasses to define the objective function.

        Parameters
        ----------
        params : dict
            Parameter values to evaluate.

        Returns
        -------
        float
            Raw objective function value (before modifiers/direction).
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _objective(self, params)")

    def _default_search_space(self) -> Dict[str, Any]:
        """Build the default search space for this function.

        Override in subclasses to provide a search space definition.
        Called by the search_space property with validation.

        Returns
        -------
        dict
            Search space mapping parameter names to value arrays.

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
        """
        raise NotImplementedError("'_default_search_space' must be implemented")

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space for this function (read-only public API)."""
        return self._default_search_space()

    # =========================================================================
    # Accessor Properties (lazy-cached)
    # =========================================================================

    @property
    def spec(self):
        """Function characteristics (SpecAccessor)."""
        # Guard: spec may be accessed before __init__ completes (e.g., BBOB
        # reads func_id in its __init__ before calling super().__init__).
        try:
            accessor = self._spec_accessor
        except AttributeError:
            accessor = None
        if accessor is None:
            from ._accessors import SpecAccessor

            accessor = SpecAccessor(self)
            try:
                self._spec_accessor = accessor
            except AttributeError:
                pass  # __init__ hasn't set up slots yet
        return accessor

    @property
    def data(self):
        """Evaluation data (DataAccessor)."""
        if self._data_accessor is None:
            from ._accessors import DataAccessor

            self._data_accessor = DataAccessor(self)
        return self._data_accessor

    @property
    def callbacks(self):
        """Callback management (CallbackAccessor)."""
        if self._callbacks_accessor is None:
            from ._accessors import CallbackAccessor

            self._callbacks_accessor = CallbackAccessor(self)
        return self._callbacks_accessor

    @property
    def modifiers(self):
        """Modifier management (ModifierAccessor)."""
        if self._modifiers_accessor is None:
            from ._accessors import ModifierAccessor

            self._modifiers_accessor = ModifierAccessor(self)
        return self._modifiers_accessor

    @property
    def memory(self):
        """Memory cache management (MemoryAccessor)."""
        if self._memory_accessor is None:
            from ._accessors import MemoryAccessor

            self._memory_accessor = MemoryAccessor(self)
        return self._memory_accessor

    @property
    def errors(self):
        """Error handler management (ErrorAccessor)."""
        if self._errors_accessor is None:
            from ._accessors import ErrorAccessor

            self._errors_accessor = ErrorAccessor(self)
        return self._errors_accessor

    @property
    def meta(self):
        """Function metadata (MetaAccessor)."""
        if self._meta_accessor is None:
            from ._accessors import MetaAccessor

            self._meta_accessor = MetaAccessor(self)
        return self._meta_accessor

    @property
    def plot(self):
        """Access plotting methods for this function."""
        from surfaces._visualize._accessor import PlotAccessor

        return PlotAccessor(self)

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

        if self._memory_enabled:
            cache_key = self._params_to_cache_key(params)
            if cache_key in self._memory_cache:
                result = self._memory_cache[cache_key]
                if self.collect_data or self._callbacks:
                    self._record_evaluation(params, result, from_cache=True)
                return result

        start_time = time.perf_counter()
        result = self._evaluate(params)
        elapsed_time = time.perf_counter() - start_time

        if self._memory_enabled:
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
            raw_value = self._objective(params)
        except Exception as e:
            if self._error_handlers is not None:
                for exc_type, return_value in self._error_handlers.items():
                    if exc_type is ... or isinstance(e, exc_type):
                        return return_value
            raise

        # Apply modifiers if configured
        if self._modifiers:
            context = {
                "evaluation_count": self._n_evaluations,
                "best_score": self._best_score,
                "search_data": self._search_data,
            }
            for modifier in self._modifiers:
                raw_value = modifier.apply(raw_value, params, context)

        if self.objective == "maximize":
            return -raw_value
        return raw_value

    # =========================================================================
    # Data Recording (inlined from DataCollectionMixin)
    # =========================================================================

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
            self._n_evaluations += 1
            self._search_data.append(record)

            if not from_cache:
                self._total_time += elapsed_time

            is_better = (
                self._best_score is None
                or (self.objective == "minimize" and score < self._best_score)
                or (self.objective == "maximize" and score > self._best_score)
            )
            if is_better:
                self._best_score = score
                self._best_params = params.copy()

        for callback in self._callbacks:
            callback(record)

    # =========================================================================
    # Evaluation Variants
    # =========================================================================

    def pure(
        self,
        params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None,
        **kwargs,
    ) -> float:
        """Evaluate the function without modifiers.

        Returns the true (deterministic) function value, bypassing any
        configured modifiers. Does not update search_data, n_evaluations,
        or callbacks. Ignores memory caching.

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
        """
        params = self._normalize_input(params, **kwargs)
        raw_value = self._objective(params)
        if self.objective == "maximize":
            return -raw_value
        return raw_value

    # =========================================================================
    # Reset Methods
    # =========================================================================

    def reset(self) -> None:
        """Reset all state including collected data and memory cache."""
        self.data.reset()
        self.memory.reset()

    # =========================================================================
    # Batch Evaluation
    # =========================================================================

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Compute the objective for a batch of parameter sets.

        Override this method in subclasses to provide vectorized evaluation.

        Parameters
        ----------
        X : ArrayLike
            2D array of shape (n_points, n_dim).

        Returns
        -------
        ArrayLike
            1D array of shape (n_points,) with objective values.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support vectorized batch evaluation. "
            "Implement _batch_objective(X) to enable this feature."
        )

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
