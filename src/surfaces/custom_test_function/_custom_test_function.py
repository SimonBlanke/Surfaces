# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CustomTestFunction - User-defined objective with rich features."""

from __future__ import annotations

import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

if TYPE_CHECKING:
    from ._namespaces import (
        AnalysisNamespace,
        PlotNamespace,
        StorageNamespace,
        SurrogateNamespace,
    )
    from .storage import Storage


class CustomTestFunction:
    """User-defined objective function with analysis, visualization, and persistence.

    Wraps any callable as a test function, providing:
    - Flexible input formats (dict, array, kwargs)
    - Automatic data collection and tracking
    - Memory caching for expensive evaluations
    - Analysis tools via `.analysis` namespace
    - Visualization via `.plot` namespace
    - Surrogate modeling via `.surrogate` namespace
    - Persistence and checkpointing (planned)

    Parameters
    ----------
    objective_fn : callable
        The objective function to evaluate. Must accept a dict of parameters
        and return a float. Signature: ``fn(params: dict) -> float``
    search_space : dict
        Search space definition. Can be:
        - Dict mapping param names to (min, max) tuples
        - Dict mapping param names to arrays of values
    resolution : int, default=100
        Number of points per dimension when bounds tuples are provided.
    experiment : str, optional
        Name for this experiment. Enables persistence and checkpointing.
    tags : list of str, optional
        Tags for filtering and organizing experiments.
    metadata : dict, optional
        Additional metadata to store with the experiment.
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    memory : bool, default=False
        If True, caches evaluated positions to avoid recomputation.
    collect_data : bool, default=True
        If True, collects evaluation history in search_data.
    storage : Storage, optional
        Storage backend for persistence. If None, data is kept in memory only.
        Built-in options: InMemoryStorage, SQLiteStorage.
        Custom backends can implement the Storage protocol.
    resume : bool, default=True
        If True and storage is provided, resume from existing experiment data.

    Attributes
    ----------
    n_evaluations : int
        Number of function evaluations performed.
    search_data : list of dict
        History of all evaluations (params + score).
    best_score : float or None
        Best score found (respects objective direction).
    best_params : dict or None
        Parameters that achieved the best score.
    total_time : float
        Cumulative time spent in evaluations (seconds).

    Examples
    --------
    Basic usage:

    >>> def sphere(params):
    ...     return sum(v**2 for v in params.values())
    >>>
    >>> func = CustomTestFunction(
    ...     objective_fn=sphere,
    ...     search_space={"x": (-5, 5), "y": (-5, 5)},
    ... )
    >>> func(x=0, y=0)
    0

    With experiment tracking:

    >>> func = CustomTestFunction(
    ...     objective_fn=sphere,
    ...     search_space={"x": (-5, 5), "y": (-5, 5)},
    ...     experiment="sphere-optimization",
    ...     tags=["test", "2d"],
    ... )

    Analysis after optimization:

    >>> # After running optimization...
    >>> func.analysis.summary()
    >>> importance = func.analysis.parameter_importance()

    Visualization:

    >>> func.plot.history()
    >>> func.plot.contour("x", "y")
    """

    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: Dict[str, Union[Tuple[float, float], np.ndarray, List]],
        resolution: int = 100,
        experiment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        objective: str = "minimize",
        memory: bool = False,
        collect_data: bool = True,
        storage: Optional["Storage"] = None,
        resume: bool = True,
    ) -> None:
        # Validate objective
        if objective not in ("minimize", "maximize"):
            raise ValueError(f"objective must be 'minimize' or 'maximize', got '{objective}'")

        # Core attributes
        self._objective_fn = objective_fn
        self._search_space = self._normalize_search_space(search_space, resolution)
        self._resolution = resolution
        self.objective = objective
        self.memory = memory
        self.collect_data = collect_data

        # Experiment metadata
        self.experiment = experiment
        self.tags = tags or []
        self.metadata = metadata or {}

        # State
        self._memory_cache: Dict[tuple, float] = {}
        self.n_evaluations: int = 0
        self.search_data: List[Dict[str, Any]] = []
        self.best_score: Optional[float] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.total_time: float = 0.0

        # Storage
        self._storage = storage

        # Resume from storage if available
        if resume and self._storage is not None:
            self._load_from_storage()

        # Lazy-initialized namespaces
        self._analysis: Optional[AnalysisNamespace] = None
        self._plot: Optional[PlotNamespace] = None
        self._storage_ns: Optional[StorageNamespace] = None
        self._surrogate: Optional[SurrogateNamespace] = None

    # =========================================================================
    # Namespace Properties (Lazy Loading)
    # =========================================================================

    @property
    def analysis(self) -> "AnalysisNamespace":
        """Analysis tools for understanding the optimization landscape.

        Provides methods for:
        - Parameter importance analysis
        - Convergence detection
        - Landscape characterization
        - Search space refinement suggestions

        Returns
        -------
        AnalysisNamespace
            Namespace with analysis methods.

        Examples
        --------
        >>> func.analysis.summary()
        >>> importance = func.analysis.parameter_importance()
        >>> func.analysis.convergence()
        """
        if self._analysis is None:
            from ._namespaces import AnalysisNamespace

            self._analysis = AnalysisNamespace(self)
        return self._analysis

    @property
    def plot(self) -> "PlotNamespace":
        """Visualization tools for the optimization results.

        Provides methods for:
        - Optimization history plots
        - Parameter importance bar charts
        - 2D contour/heatmap plots
        - 3D surface plots
        - Parallel coordinates

        Returns
        -------
        PlotNamespace
            Namespace with plotting methods.

        Examples
        --------
        >>> func.plot.history()
        >>> func.plot.contour("x", "y")
        >>> func.plot.importance()
        """
        if self._plot is None:
            from ._namespaces import PlotNamespace

            self._plot = PlotNamespace(self)
        return self._plot

    @property
    def surrogate(self) -> "SurrogateNamespace":
        """Surrogate modeling tools.

        Provides methods for:
        - Fitting surrogate models (GP, RF, etc.)
        - Fast predictions without expensive evaluations
        - Uncertainty quantification
        - Active learning suggestions

        Returns
        -------
        SurrogateNamespace
            Namespace with surrogate modeling methods.

        Examples
        --------
        >>> func.surrogate.fit(method="gaussian_process")
        >>> prediction = func.surrogate.predict({"x": 0.5, "y": 0.5})
        >>> uncertainty = func.surrogate.uncertainty({"x": 0.5, "y": 0.5})
        """
        if self._surrogate is None:
            from ._namespaces import SurrogateNamespace

            self._surrogate = SurrogateNamespace(self)
        return self._surrogate

    @property
    def storage(self) -> "StorageNamespace":
        """Storage and persistence operations.

        Provides methods for:
        - Checkpointing (save/load state)
        - Querying stored evaluations
        - Managing experiment data

        Note: Most methods require a storage backend to be configured.
        Without a storage backend, methods will raise RuntimeError.

        Returns
        -------
        StorageNamespace
            Namespace with storage and persistence methods.

        Examples
        --------
        >>> func.storage.save_checkpoint()
        >>> func.storage.load_checkpoint()
        >>> best = func.storage.query(order_by="score", limit=10)
        """
        if self._storage_ns is None:
            from ._namespaces import StorageNamespace

            self._storage_ns = StorageNamespace(self)
        return self._storage_ns

    # =========================================================================
    # Search Space
    # =========================================================================

    def _normalize_search_space(
        self,
        search_space: Dict[str, Union[Tuple[float, float], np.ndarray, List]],
        resolution: int,
    ) -> Dict[str, np.ndarray]:
        """Convert search space to dict of arrays."""
        normalized = {}
        for name, value in search_space.items():
            if isinstance(value, np.ndarray):
                normalized[name] = value
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                min_val, max_val = value
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    # Bounds tuple: (min, max)
                    normalized[name] = np.linspace(min_val, max_val, resolution)
                else:
                    # List of values
                    normalized[name] = np.array(value)
            elif isinstance(value, list):
                normalized[name] = np.array(value)
            else:
                raise ValueError(
                    f"Invalid search space for '{name}': expected array, list, or "
                    f"(min, max) tuple, got {type(value).__name__}"
                )
        return normalized

    @property
    def search_space(self) -> Dict[str, np.ndarray]:
        """Search space for this function."""
        return self._search_space

    @property
    def n_dim(self) -> int:
        """Number of dimensions."""
        return len(self._search_space)

    @property
    def param_names(self) -> List[str]:
        """Parameter names in sorted order."""
        return sorted(self._search_space.keys())

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Parameter bounds as (min, max) tuples."""
        return {name: (arr.min(), arr.max()) for name, arr in self._search_space.items()}

    # =========================================================================
    # Call Interface
    # =========================================================================

    def __call__(
        self,
        params: Optional[Union[Dict[str, Any], np.ndarray, List, Tuple]] = None,
        **kwargs,
    ) -> float:
        """Evaluate the objective function.

        Parameters
        ----------
        params : dict, array, list, or tuple, optional
            Parameter values. Can be:
            - Dict: {"x": 1.0, "y": 2.0}
            - Array/List: [1.0, 2.0] (mapped to sorted param names)
        **kwargs
            Parameters as keyword arguments.

        Returns
        -------
        float
            The objective function value.

        Examples
        --------
        >>> func({"x": 1.0, "y": 2.0})
        >>> func([1.0, 2.0])
        >>> func(x=1.0, y=2.0)
        """
        params = self._normalize_input(params, **kwargs)

        # Check memory cache
        if self.memory:
            cache_key = self._params_to_cache_key(params)
            if cache_key in self._memory_cache:
                score = self._memory_cache[cache_key]
                if self.collect_data:
                    self._record_evaluation(params, score, from_cache=True)
                return score

        # Evaluate
        start_time = time.perf_counter()
        raw_score = self._objective_fn(params)
        elapsed = time.perf_counter() - start_time

        # Apply objective direction
        score = -raw_score if self.objective == "maximize" else raw_score

        # Cache result
        if self.memory:
            cache_key = self._params_to_cache_key(params)
            self._memory_cache[cache_key] = score

        # Record
        if self.collect_data:
            self._record_evaluation(params, score, elapsed)

        return score

    def _normalize_input(
        self,
        params: Optional[Union[Dict[str, Any], np.ndarray, List, Tuple]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert any input format to dict."""
        if isinstance(params, (np.ndarray, list, tuple)):
            param_names = self.param_names
            if len(params) != len(param_names):
                raise ValueError(f"Expected {len(param_names)} values, got {len(params)}")
            return {name: params[i] for i, name in enumerate(param_names)}

        if params is None:
            params = {}
        return {**params, **kwargs}

    def _params_to_cache_key(self, params: Dict[str, Any]) -> Tuple:
        """Convert params to hashable cache key."""
        return tuple(params[k] for k in sorted(params.keys()))

    def _record_evaluation(
        self,
        params: Dict[str, Any],
        score: float,
        elapsed: float = 0.0,
        from_cache: bool = False,
    ) -> None:
        """Record an evaluation."""
        self.n_evaluations += 1
        evaluation = {**params, "score": score}
        self.search_data.append(evaluation)

        if not from_cache:
            self.total_time += elapsed

        # Update best
        is_better = self.best_score is None or (
            (self.objective == "minimize" and score < self.best_score)
            or (self.objective == "maximize" and score > self.best_score)
        )
        if is_better:
            self.best_score = score
            self.best_params = params.copy()

        # Persist to storage
        if self._storage is not None:
            self._storage.save_evaluation(evaluation)

    # =========================================================================
    # Data Access
    # =========================================================================

    def get_data_as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get evaluation data as numpy arrays.

        Returns
        -------
        X : ndarray of shape (n_evaluations, n_dim)
            Parameter values.
        y : ndarray of shape (n_evaluations,)
            Scores.
        """
        if not self.search_data:
            return np.array([]).reshape(0, self.n_dim), np.array([])

        param_names = self.param_names
        X = np.array([[d[name] for name in param_names] for d in self.search_data])
        y = np.array([d["score"] for d in self.search_data])
        return X, y

    # =========================================================================
    # State Management
    # =========================================================================

    def reset(self) -> None:
        """Reset all state including evaluations and cache."""
        self.n_evaluations = 0
        self.search_data = []
        self.best_score = None
        self.best_params = None
        self.total_time = 0.0
        self._memory_cache = {}

    def reset_cache(self) -> None:
        """Clear only the memory cache."""
        self._memory_cache = {}

    # =========================================================================
    # Storage & Persistence (Internal)
    # =========================================================================

    def _load_from_storage(self) -> None:
        """Load existing data from storage."""
        if self._storage is None:
            return

        # Load evaluations
        evaluations = self._storage.load_evaluations()
        if evaluations:
            self.search_data = []
            for evaluation in evaluations:
                # Remove internal metadata before adding to search_data
                record = {k: v for k, v in evaluation.items() if not k.startswith("_")}
                self.search_data.append(record)

            self.n_evaluations = len(evaluations)

            # Rebuild best tracking
            for record in self.search_data:
                score = record["score"]
                params = {k: v for k, v in record.items() if k != "score"}
                is_better = self.best_score is None or (
                    (self.objective == "minimize" and score < self.best_score)
                    or (self.objective == "maximize" and score > self.best_score)
                )
                if is_better:
                    self.best_score = score
                    self.best_params = params

        # Load state/checkpoint if available
        state = self._storage.load_state()
        if state:
            self.total_time = state.get("total_time", 0.0)
            if "memory_cache" in state:
                self._memory_cache = state["memory_cache"]

    def close(self) -> None:
        """Close the storage connection and release resources.

        Call this when done with the function to ensure proper cleanup.
        Can also use as context manager for automatic cleanup.

        Equivalent to ``func.storage.close()``.
        """
        self.storage.close()

    def __enter__(self) -> "CustomTestFunction":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing storage."""
        self.close()

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:
        parts = [
            "CustomTestFunction(",
            f"  n_dim={self.n_dim},",
            f"  params={self.param_names},",
            f"  objective='{self.objective}',",
            f"  n_evaluations={self.n_evaluations},",
        ]
        if self.experiment:
            parts.append(f"  experiment='{self.experiment}',")
        if self.best_score is not None:
            parts.append(f"  best_score={self.best_score:.6g},")
        parts.append(")")
        return "\n".join(parts)
