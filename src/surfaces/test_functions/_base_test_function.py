# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import dataclasses
import functools
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from surfaces._array_utils import ArrayLike, is_array_like
from surfaces.modifiers import BaseModifier

from ._function_spec import FunctionSpec, MetaSpec, resolve_function_spec, resolve_meta_spec


def _check_dependencies_after_init(init_func):
    """Call ``_check_dependencies()`` after ``__init__`` completes."""

    @functools.wraps(init_func)
    def wrapper(self, *args, **kwargs):
        result = init_func(self, *args, **kwargs)
        self._check_dependencies()
        return result

    return wrapper


class BaseTestFunction:
    """Base class for all test functions in the Surfaces library.

    This is the generic root class providing the template method pattern
    for evaluation, memory caching, data collection, callbacks, and
    error handling. Subclasses should inherit from one of the two
    intermediate bases:

    - :class:`BaseSingleObjectiveTestFunction` for scalar objectives
    - :class:`BaseMultiObjectiveTestFunction` for vector objectives

    Parameters
    ----------
    objective : str, list, or None
        Optimization direction. Interpreted by subclasses:
        single-objective uses ``"minimize"``/``"maximize"``;
        multi-objective accepts a list of per-objective directions.
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
        raw_spec = cls.__dict__.get("_spec")
        raw_meta = cls.__dict__.get("_meta")

        # Auto-derive name if not explicitly defined
        if "name" not in cls.__dict__:
            spec_name = raw_spec.get("name") if hasattr(raw_spec, "get") else None
            meta_name = raw_meta.get("name") if hasattr(raw_meta, "get") else None
            if meta_name is not None:
                cls.name = meta_name
            elif spec_name is not None:
                cls.name = spec_name
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
            meta_slug = raw_meta.get("slug") if hasattr(raw_meta, "get") else None
            legacy_meta_slug = raw_meta.get("_name_") if hasattr(raw_meta, "get") else None
            if meta_slug is not None:
                cls._name_ = meta_slug
            elif legacy_meta_slug is not None:
                cls._name_ = legacy_meta_slug
            else:
                cls._name_ = cls.name.lower().replace(" ", "_")

        cls._spec = resolve_function_spec(cls, raw_spec)
        cls._meta = resolve_meta_spec(cls, raw_spec, raw_meta)
        if cls._meta.name is not None:
            cls.name = cls._meta.name
        if cls._meta.slug is not None:
            cls._name_ = cls._meta.slug

    _spec: FunctionSpec = FunctionSpec()
    _meta: MetaSpec = MetaSpec(name="Base Test Function", slug="base_test_function")

    f_global: Optional[float] = None
    x_global: Optional[np.ndarray] = None

    # Optional dependencies: {extras_group: [package, ...]}
    # Subclasses set this to declare packages checked at init time.
    _dependencies = None

    # Type alias for callbacks
    CallbackType = Union[Callable[[Dict[str, Any]], None], List[Callable[[Dict[str, Any]], None]]]

    @_check_dependencies_after_init
    def __init__(
        self,
        objective=None,
        modifiers: Optional[List[BaseModifier]] = None,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
    ):
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

        # Accessor caches (lazy-loaded). spec/meta are resolved on the fly
        # by their properties (returning dataclasses), so they are not cached.
        self._data_accessor = None
        self._callbacks_accessor = None
        self._modifiers_accessor = None
        self._memory_accessor = None
        self._errors_accessor = None

        self._active_fidelity: Optional[float] = None

    def __repr__(self) -> str:
        """Show class name, dimensionality, and optimization direction."""
        parts = []
        n_dim = getattr(self, "n_dim", None)
        if n_dim is not None:
            parts.append(f"n_dim={n_dim}")
        if self.objective is not None:
            parts.append(f"objective={self.objective!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

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

    @property
    def spec(self) -> FunctionSpec:
        """Instance-resolved function specification (a frozen FunctionSpec).

        ``type(self)._spec`` is the static class-level template. This property
        overlays the fields that genuinely vary per instance (``n_dim``,
        ``n_objectives``, ``f_global``, ``x_global``) by lifting them off the
        instance, so that ``func.spec.n_dim`` reflects this instance's value.
        It is resolved on every access rather than cached, because some
        functions (e.g. BBOB) read ``spec`` during ``__init__`` before the
        optimum has been computed, and a cached early value would go stale.
        """
        base = type(self)._spec
        overrides = {}
        for field_name in ("n_dim", "n_objectives", "f_global", "x_global"):
            # Skip fields the function does not define at all (e.g.
            # n_objectives on single-objective functions); their value comes
            # from the class spec. Deliberately NOT a getattr-with-default or
            # a bare ``except AttributeError``: those would also swallow an
            # AttributeError raised by a genuinely broken property and hide a
            # real bug. Checking class/instance definition first lets a real
            # property error propagate while still tolerating undefined fields.
            if field_name not in self.__dict__ and not hasattr(type(self), field_name):
                continue
            value = getattr(self, field_name)
            if value is not None:
                overrides[field_name] = value
        return dataclasses.replace(base, **overrides) if overrides else base

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
    def meta(self) -> MetaSpec:
        """Instance display/identity metadata (a frozen MetaSpec).

        Metadata is fully static today, so this returns the class-level
        ``MetaSpec`` resolved at class-definition time.
        """
        return type(self)._meta

    @property
    def plot(self):
        """Access plotting methods for this function."""
        from surfaces._visualize._accessor import PlotAccessor

        return PlotAccessor(self)

    def __call__(
        self,
        params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None,
        *,
        fidelity: Optional[float] = None,
        **kwargs,
    ):
        """Evaluate the objective function.

        Args:
            params: Parameter values as dict, array, list, or tuple
            fidelity: Optional fidelity level in (0, 1]. Controls evaluation
                cost vs accuracy trade-off for multi-fidelity optimization
                (e.g. Hyperband, BOHB). None means full-fidelity evaluation.
                Only supported by ML test functions; ignored by algebraic functions.
            **kwargs: Parameters as keyword arguments (only with dict input)

        Returns:
            The objective function value
        """
        if fidelity is not None:
            if not isinstance(fidelity, (int, float)):
                raise TypeError(f"fidelity must be a number, got {type(fidelity).__name__}")
            if not (0 < fidelity <= 1.0):
                raise ValueError(f"fidelity must be in (0, 1], got {fidelity}")

        params = self._normalize_input(params, **kwargs)

        if self._memory_enabled:
            cache_key = self._params_to_cache_key(params, fidelity=fidelity)
            if cache_key in self._memory_cache:
                result = self._memory_cache[cache_key]
                if self.collect_data or self._callbacks:
                    self._record_evaluation(params, result, from_cache=True, fidelity=fidelity)
                return result

        start_time = time.perf_counter()
        result = self._evaluate(params, fidelity=fidelity)
        elapsed_time = time.perf_counter() - start_time

        if self._memory_enabled:
            cache_key = self._params_to_cache_key(params, fidelity=fidelity)
            self._memory_cache[cache_key] = result

        if self.collect_data or self._callbacks:
            self._record_evaluation(params, result, elapsed_time=elapsed_time, fidelity=fidelity)

        return result

    def _normalize_input(
        self,
        params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert any input format to dict."""
        if isinstance(params, (np.ndarray, list, tuple)):
            param_names = self._array_input_param_names()
            if len(params) != len(param_names):
                raise ValueError(f"Expected {len(param_names)} values, got {len(params)}")
            return {name: params[i] for i, name in enumerate(param_names)}

        if params is None:
            params = {}
        return {**params, **kwargs}

    def _array_input_param_names(self) -> List[str]:
        """Parameter order used for 1D array/list/tuple input.

        The default preserves the historical contract: array-like scalar
        evaluations are mapped by alphabetically sorted search-space keys.
        Subclasses with a domain-specific positional order can override this.
        """
        return sorted(self.search_space.keys())

    def _params_to_cache_key(
        self, params: Dict[str, Any], fidelity: Optional[float] = None
    ) -> Tuple:
        """Convert params dict to a hashable cache key (sorted tuple of values)."""
        key = tuple(params[k] for k in sorted(params.keys()))
        if fidelity is not None:
            key = (*key, ("__fidelity__", fidelity))
        return key

    def _apply_direction(self, value):
        """Apply objective-direction transformation.

        Identity in the base class. Overridden by:
        - ``BaseSingleObjectiveTestFunction``: negate for maximize
        - ``BaseMultiObjectiveTestFunction``: per-element negation
        - ``MachineLearningFunction``: negate for minimize

        Parameters
        ----------
        value
            The evaluated value (scalar or array).

        Returns
        -------
        same type as *value*
            Direction-adjusted value.
        """
        return value

    def _compute_raw(self, params: Dict[str, Any], fidelity: Optional[float] = None):
        """Produce the raw objective value for a single evaluation.

        Default: delegate to ``_objective``; ``fidelity`` is ignored.
        Subclasses that source values differently (e.g.
        ``MachineLearningFunction`` with surrogate models and
        fidelity-based subsampling) override this hook instead of
        ``_evaluate``, so that error handling, modifiers, and direction
        stay in one place for every function family.

        Parameters
        ----------
        params : dict
            Parameter values to evaluate.
        fidelity : float or None
            Fidelity level in (0, 1], or None for full fidelity.
        """
        return self._objective(params)

    def _evaluate(self, params: Dict[str, Any], fidelity: Optional[float] = None):
        """Evaluate with error handling, modifiers, and direction.

        Pipeline: _compute_raw -> error handling -> _apply_modifiers -> _apply_direction

        Parameters
        ----------
        params : dict
            Parameter values to evaluate.
        fidelity : float or None
            Fidelity level in (0, 1]. Passed through from __call__ to
            ``_compute_raw``, which ignores it by default.
        """
        try:
            raw_value = self._compute_raw(params, fidelity=fidelity)
        except Exception as e:
            if self._error_handlers is not None:
                for exc_type, return_value in self._error_handlers.items():
                    if exc_type is ... or isinstance(e, exc_type):
                        return return_value
            raise

        if self._modifiers:
            raw_value = self._apply_modifiers(raw_value, params)

        return self._apply_direction(raw_value)

    def _apply_modifiers(self, raw_value, params):
        """Apply modifier pipeline to the raw value.

        ``raw_value`` is a scalar for single-objective functions and an
        objective vector for multi-objective functions; modifiers receive
        it unchanged in both cases.
        """
        context = {
            "evaluation_count": self._n_evaluations,
            "best_score": self._best_score,
            "search_data": self._search_data,
        }
        for modifier in self._modifiers:
            raw_value = modifier.apply(raw_value, params, context)
        return raw_value

    def _record_evaluation(
        self,
        params: Dict[str, Any],
        score,
        elapsed_time: float = 0.0,
        from_cache: bool = False,
        fidelity: Optional[float] = None,
    ) -> None:
        """Record an evaluation and invoke callbacks."""
        record = {**params, "score": score}
        if fidelity is not None:
            record["fidelity"] = fidelity

        if self.collect_data:
            self._n_evaluations += 1
            self._search_data.append(record)

            if not from_cache:
                self._total_time += elapsed_time

            self._update_best(params, score)

        for callback in self._callbacks:
            callback(record)

    def _update_best(self, params: Dict[str, Any], score) -> None:
        """Update best score/params if score is an improvement.

        No-op in the base class. Overridden by
        :class:`BaseSingleObjectiveTestFunction` for scalar comparison.
        """

    def pure(
        self,
        params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None,
        *,
        fidelity: Optional[float] = None,
        **kwargs,
    ):
        """Evaluate the function without modifiers.

        Returns the true (deterministic) function value, bypassing any
        configured modifiers. Does not update search_data, n_evaluations,
        or callbacks. Ignores memory caching.

        Parameters
        ----------
        params : dict, array, list, or tuple
            Parameter values to evaluate.
        fidelity : float or None
            Fidelity level in (0, 1] for multi-fidelity evaluation.
        **kwargs : dict
            Parameters as keyword arguments.

        Returns
        -------
        float or np.ndarray
            The true function value without modifiers, with direction applied.
        """
        params = self._normalize_input(params, **kwargs)
        # Temporarily set fidelity for ML functions that read it during _objective
        self._active_fidelity = fidelity
        try:
            raw_value = self._objective(params)
        finally:
            self._active_fidelity = None
        return self._apply_direction(raw_value)

    def reset(self) -> None:
        """Reset all state including collected data and memory cache."""
        self.data.reset()
        self.memory.reset()

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
                f"Expected array-like input with shape (n_points, n_dim), got {type(X).__name__}"
            )

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array with shape (n_points, n_dim), got {X.ndim}D array")

        if X.shape[1] != self.n_dim:
            raise ValueError(f"Expected {self.n_dim} dimensions, got {X.shape[1]}")

        result = self._batch_objective(X)
        return self._apply_direction(result)
