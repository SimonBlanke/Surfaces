# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Storage namespace for CustomTestFunction persistence operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .._custom_test_function import CustomTestFunction
    from ..storage import Storage as StorageBackend


class StorageNamespace:
    """Namespace for storage and persistence operations.

    Provides methods for:
    - Checkpointing (save/load state)
    - Querying stored evaluations
    - Managing experiment data

    This namespace is only functional when a storage backend is configured.
    Without a storage backend, methods will raise RuntimeError.

    Examples
    --------
    >>> from surfaces.custom_test_function import CustomTestFunction, SQLiteStorage
    >>>
    >>> storage = SQLiteStorage("./exp.db", experiment="my-exp")
    >>> func = CustomTestFunction(
    ...     objective_fn=my_objective,
    ...     search_space={"x": (-5, 5)},
    ...     storage=storage,
    ... )
    >>>
    >>> # After some evaluations...
    >>> func.storage.save_checkpoint()
    >>>
    >>> # Query best results
    >>> best = func.storage.query(order_by="score", limit=10)
    >>>
    >>> # Later, in a new session
    >>> func.storage.load_checkpoint()
    """

    def __init__(self, func: "CustomTestFunction") -> None:
        self._func = func

    @property
    def backend(self) -> Optional["StorageBackend"]:
        """The underlying storage backend, if configured."""
        return self._func._storage

    @property
    def is_configured(self) -> bool:
        """Whether a storage backend is configured."""
        return self._func._storage is not None

    @property
    def experiment(self) -> Optional[str]:
        """The experiment name from the storage backend."""
        if self._func._storage is None:
            return None
        return self._func._storage.experiment

    def _require_storage(self) -> None:
        """Raise if no storage backend configured."""
        if self._func._storage is None:
            raise RuntimeError(
                "No storage backend configured. "
                "Pass a Storage instance to CustomTestFunction constructor."
            )

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def save_checkpoint(self) -> None:
        """Save current state to storage for later resumption.

        Saves:
        - Current best score and parameters
        - Total evaluation time
        - Memory cache (if memory=True)
        - Custom metadata

        Raises
        ------
        RuntimeError
            If no storage backend is configured.

        Examples
        --------
        >>> func.storage.save_checkpoint()
        >>> # Later, in a new session with same storage
        >>> func.storage.load_checkpoint()
        """
        self._require_storage()

        state = {
            "n_evaluations": self._func.n_evaluations,
            "best_score": self._func.best_score,
            "best_params": self._func.best_params,
            "total_time": self._func.total_time,
            "memory_cache": self._func._memory_cache,
            "metadata": self._func.metadata,
            "objective": self._func.objective,
        }
        self._func._storage.save_state(state)

    def load_checkpoint(self) -> bool:
        """Load checkpoint from storage.

        This is called automatically on initialization if resume=True.
        Use this method to manually reload state.

        Returns
        -------
        bool
            True if checkpoint was loaded, False if no checkpoint exists.

        Raises
        ------
        RuntimeError
            If no storage backend is configured.
        """
        self._require_storage()

        state = self._func._storage.load_state()
        if state is None:
            return False

        self._func.total_time = state.get("total_time", 0.0)
        if "memory_cache" in state:
            self._func._memory_cache = state["memory_cache"]

        return True

    # =========================================================================
    # Data Access
    # =========================================================================

    def query(
        self,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query stored evaluations with filtering and ordering.

        Parameters
        ----------
        filter_fn : callable, optional
            Function that takes an evaluation dict and returns True to include.
        order_by : str, optional
            Field name to order by. Prefix with "-" for descending.
            Common values: "score", "-score", "_timestamp".
        limit : int, optional
            Maximum number of results to return.

        Returns
        -------
        list of dict
            Filtered and ordered evaluations.

        Raises
        ------
        RuntimeError
            If no storage backend is configured.

        Examples
        --------
        >>> # Get top 10 evaluations by score
        >>> best = func.storage.query(order_by="score", limit=10)
        >>>
        >>> # Get evaluations where x > 0
        >>> positive_x = func.storage.query(filter_fn=lambda e: e["x"] > 0)
        >>>
        >>> # Get most recent evaluations
        >>> recent = func.storage.query(order_by="-_timestamp", limit=5)
        """
        self._require_storage()
        return self._func._storage.query(
            filter_fn=filter_fn,
            order_by=order_by,
            limit=limit,
        )

    def load_evaluations(self) -> List[Dict[str, Any]]:
        """Load all evaluations from storage.

        Returns
        -------
        list of dict
            All stored evaluations for this experiment.

        Raises
        ------
        RuntimeError
            If no storage backend is configured.
        """
        self._require_storage()
        return self._func._storage.load_evaluations()

    def count(self) -> int:
        """Count stored evaluations.

        Returns
        -------
        int
            Number of evaluations in storage.

        Raises
        ------
        RuntimeError
            If no storage backend is configured.
        """
        self._require_storage()
        return len(self._func._storage.load_evaluations())

    # =========================================================================
    # Experiment Management
    # =========================================================================

    def delete(self) -> None:
        """Delete all stored data for this experiment.

        This permanently removes all evaluations and checkpoints from storage.
        Also resets the in-memory state of the function.

        Use with caution - this operation cannot be undone.

        Raises
        ------
        RuntimeError
            If no storage backend is configured.
        """
        self._require_storage()
        self._func._storage.delete_experiment()
        self._func.reset()

    def list_experiments(self) -> List[str]:
        """List all experiments in the storage backend.

        Returns
        -------
        list of str
            Experiment names/identifiers.

        Raises
        ------
        RuntimeError
            If no storage backend is configured.
        NotImplementedError
            If the storage backend doesn't support listing experiments.
        """
        self._require_storage()
        return self._func._storage.list_experiments()

    def exists(self) -> bool:
        """Check if this experiment has stored data.

        Returns
        -------
        bool
            True if the experiment has evaluations in storage.

        Raises
        ------
        RuntimeError
            If no storage backend is configured.
        """
        self._require_storage()
        return self._func._storage.experiment_exists()

    # =========================================================================
    # Connection Management
    # =========================================================================

    def close(self) -> None:
        """Close the storage connection and release resources.

        Call this when done with the function to ensure proper cleanup.
        The CustomTestFunction context manager calls this automatically.
        """
        if self._func._storage is not None:
            self._func._storage.close()

    def __repr__(self) -> str:
        if self._func._storage is None:
            return "StorageNamespace(backend=None)"
        return f"StorageNamespace(backend={self._func._storage!r})"
