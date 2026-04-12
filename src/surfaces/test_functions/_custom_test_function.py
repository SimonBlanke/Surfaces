# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""User-defined test function with full Surfaces infrastructure."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from ._base_single_objective import BaseSingleObjectiveTestFunction

if TYPE_CHECKING:
    from ._custom_namespaces import (
        AnalysisNamespace,
        PlotNamespace,
        StorageNamespace,
        SurrogateNamespace,
    )
    from ._custom_storage import Storage


class CustomTestFunction(BaseSingleObjectiveTestFunction):
    """User-defined test function with full Surfaces infrastructure.

    Wraps any callable as a Surfaces test function, providing all
    inherited features (modifiers, memory, callbacks, batch evaluation,
    plotting, error handling) plus custom extensions for analysis,
    surrogate modeling, and persistent storage.

    Parameters
    ----------
    objective_fn : callable
        The objective function to evaluate. Must accept a dict of
        parameters and return a float.
    search_space : dict
        Parameter definitions. Values can be numpy arrays, lists,
        or ``(min, max)`` bounds tuples (converted via ``resolution``).
    resolution : int, default=100
        Points per dimension when bounds tuples are provided.
    global_optimum : dict, optional
        Known optimum with keys ``"position"`` (dict) and ``"score"`` (float).
    objective : str, default="minimize"
        Either ``"minimize"`` or ``"maximize"``.
    modifiers : list of BaseModifier, optional
        Modifiers applied to evaluations (noise, delay, etc.).
    memory : bool, default=False
        Cache evaluated positions to avoid recomputation.
    collect_data : bool, default=True
        Track evaluation history, best score, and timing.
    callbacks : callable or list of callables, optional
        Called after each evaluation with the record dict.
    catch_errors : dict, optional
        Map exception types to fallback return values.
    experiment : str, optional
        Experiment name for persistence and tracking.
    tags : list of str, optional
        Tags for organizing experiments.
    metadata : dict, optional
        Arbitrary metadata stored with the experiment.
    storage : Storage, optional
        Persistence backend (InMemoryStorage, SQLiteStorage, or custom).
    resume : bool, default=True
        If True and storage is provided, restore state from storage.

    Examples
    --------
    Minimal usage (no search space needed):

    >>> from surfaces import CustomTestFunction
    >>>
    >>> func = CustomTestFunction(lambda p: p["x"]**2 + p["y"]**2)
    >>> func(x=1, y=2)
    5

    With search space (enables plotting and surrogate features):

    >>> func = CustomTestFunction(
    ...     lambda p: p["x"]**2 + p["y"]**2,
    ...     search_space={"x": (-5, 5), "y": (-5, 5)},
    ... )
    >>> func.plot.surface()  # Plotly 3D surface
    """

    _spec: Dict[str, Any] = {
        "name": "CustomTestFunction",
        "continuous": None,
        "differentiable": None,
        "convex": None,
        "separable": None,
        "unimodal": None,
        "scalable": None,
    }

    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: Optional[Dict[str, Union[np.ndarray, Tuple[float, float], List]]] = None,
        resolution: int = 100,
        global_optimum: Optional[Dict[str, Any]] = None,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        experiment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        storage: Optional["Storage"] = None,
        resume: bool = True,
    ) -> None:
        self._user_objective_fn = objective_fn
        if search_space is not None:
            self._search_space_data = self._normalize_search_space(search_space, resolution)
        else:
            self._search_space_data = None
        self._resolution = resolution

        if global_optimum is not None:
            self.x_global = global_optimum.get("position")
            self.f_global = global_optimum.get("score")

        n_dim = len(self._search_space_data) if self._search_space_data is not None else None
        self._spec = {**self._spec, "n_dim": n_dim}

        self.experiment = experiment
        self.tags = tags or []
        self.metadata = metadata or {}
        self._storage = storage

        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

        # Persist every evaluation to storage via the callback mechanism
        if self._storage is not None:
            self._callbacks.append(self._persist_to_storage)

        if resume and self._storage is not None:
            self._load_from_storage()

        # Lazy-initialized custom namespaces
        self._analysis_ns: Optional[AnalysisNamespace] = None
        self._custom_plot_ns: Optional[PlotNamespace] = None
        self._surrogate_ns: Optional[SurrogateNamespace] = None
        self._storage_ns: Optional[StorageNamespace] = None

    def _objective(self, params: Dict[str, Any]) -> float:
        """Delegate to user-provided objective function."""
        return self._user_objective_fn(params)

    def _default_search_space(self) -> Dict[str, np.ndarray]:
        """Return the user-provided search space."""
        if self._search_space_data is None:
            raise ValueError(
                "No search space defined. Pass search_space to the constructor "
                "to use features that require it (plotting, surrogate suggestions, "
                "benchmark runner)."
            )
        return self._search_space_data

    # Convenience properties that delegate to the inherited DataAccessor.
    # These keep the custom namespaces (analysis, surrogate, etc.) working
    # without modification, and provide a simpler interface for users who
    # don't need the full accessor pattern.

    @property
    def n_evaluations(self) -> int:
        """Number of evaluations performed."""
        return self.data.n_evaluations

    @property
    def search_data(self) -> List[Dict[str, Any]]:
        """History of all evaluations (params + score)."""
        return self.data.search_data

    @property
    def best_score(self) -> Optional[float]:
        """Best score found (respects objective direction)."""
        return self.data.best_score

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Parameters that achieved the best score."""
        return self.data.best_params

    @property
    def total_time(self) -> float:
        """Cumulative time spent in evaluations (seconds)."""
        return self.data.total_time

    @property
    def n_dim(self) -> Optional[int]:
        """Number of dimensions, or None if no search space is defined."""
        if self._search_space_data is None:
            return None
        return len(self._search_space_data)

    @property
    def param_names(self) -> Optional[List[str]]:
        """Parameter names in sorted order, or None if no search space."""
        if self._search_space_data is None:
            return None
        return sorted(self._search_space_data.keys())

    @property
    def bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Parameter bounds as ``(min, max)`` tuples, or None if no search space."""
        if self._search_space_data is None:
            return None
        return {
            name: (float(arr.min()), float(arr.max()))
            for name, arr in self._search_space_data.items()
        }

    # Custom namespaces (not available on other test functions)

    @property
    def analysis(self) -> "AnalysisNamespace":
        """Analysis tools for optimization results.

        Provides parameter importance, convergence detection,
        and search space refinement suggestions.
        """
        if self._analysis_ns is None:
            from ._custom_namespaces import AnalysisNamespace

            self._analysis_ns = AnalysisNamespace(self)
        return self._analysis_ns

    @property
    def custom_plots(self) -> "PlotNamespace":
        """Data-driven plots for optimization history.

        Provides history scatter, parameter importance bars,
        contour from collected data, and parallel coordinates.
        Uses matplotlib. For function-landscape plots (surface,
        contour, heatmap), use the inherited ``func.plot`` accessor.
        """
        if self._custom_plot_ns is None:
            from ._custom_namespaces import PlotNamespace

            self._custom_plot_ns = PlotNamespace(self)
        return self._custom_plot_ns

    @property
    def surrogate(self) -> "SurrogateNamespace":
        """Surrogate modeling tools.

        Fit GP, RandomForest, or GradientBoosting models to collected
        evaluations. Predict, quantify uncertainty, and suggest next
        evaluation points via Expected Improvement.
        """
        if self._surrogate_ns is None:
            from ._custom_namespaces import SurrogateNamespace

            self._surrogate_ns = SurrogateNamespace(self)
        return self._surrogate_ns

    @property
    def storage(self) -> "StorageNamespace":
        """Storage and persistence operations.

        Save/load checkpoints, query stored evaluations, and
        manage experiment data. Requires a storage backend.
        """
        if self._storage_ns is None:
            from ._custom_namespaces import StorageNamespace

            self._storage_ns = StorageNamespace(self)
        return self._storage_ns

    # Custom methods

    def get_data_as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get evaluation data as numpy arrays.

        Returns
        -------
        X : ndarray of shape (n_evaluations, n_dim)
            Parameter values.
        y : ndarray of shape (n_evaluations,)
            Scores.
        """
        data = self.data.search_data
        if not data:
            n_dim = self.n_dim or 0
            return np.array([]).reshape(0, n_dim), np.array([])

        # Derive param names from data if no search space was defined
        names = self.param_names
        if names is None:
            names = sorted(k for k in data[0] if k != "score")

        X = np.array([[d[name] for name in names] for d in data])
        y = np.array([d["score"] for d in data])
        return X, y

    def reset(self) -> None:
        """Reset all state: evaluations, best tracking, and memory cache."""
        self.data.reset()
        self.memory.reset()

    def reset_cache(self) -> None:
        """Clear only the memory cache, preserving evaluation history."""
        self.memory.reset()

    def close(self) -> None:
        """Close the storage connection and release resources."""
        if self._storage_ns is not None:
            self._storage_ns.close()
        elif self._storage is not None:
            self._storage.close()

    def __enter__(self) -> "CustomTestFunction":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # Internal

    def _persist_to_storage(self, record: Dict[str, Any]) -> None:
        """Callback that saves each evaluation to the storage backend."""
        if self._storage is not None:
            self._storage.save_evaluation(record)

    def _load_from_storage(self) -> None:
        """Restore state from the storage backend."""
        if self._storage is None:
            return

        evaluations = self._storage.load_evaluations()
        if evaluations:
            for evaluation in evaluations:
                record = {k: v for k, v in evaluation.items() if not k.startswith("_")}
                self._search_data.append(record)
                self._n_evaluations += 1

                score = record["score"]
                params = {k: v for k, v in record.items() if k != "score"}
                self._update_best(params, score)

        state = self._storage.load_state()
        if state:
            self._total_time = state.get("total_time", 0.0)
            if "memory_cache" in state:
                self._memory_cache = state["memory_cache"]

    @staticmethod
    def _normalize_search_space(
        search_space: Dict[str, Union[np.ndarray, Tuple[float, float], List]],
        resolution: int,
    ) -> Dict[str, np.ndarray]:
        """Convert search space definitions to arrays."""
        normalized = {}
        for name, value in search_space.items():
            if isinstance(value, np.ndarray):
                normalized[name] = value
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                min_val, max_val = value
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    normalized[name] = np.linspace(min_val, max_val, resolution)
                else:
                    normalized[name] = np.array(value)
            elif isinstance(value, list):
                normalized[name] = np.array(value)
            else:
                raise ValueError(
                    f"Invalid search space for '{name}': expected array, list, or "
                    f"(min, max) tuple, got {type(value).__name__}"
                )
        return normalized

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
