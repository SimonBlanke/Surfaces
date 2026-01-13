# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Storage protocol for CustomTestFunction persistence.

This module defines the Storage protocol that all storage backends must implement.
Users can create custom storage backends by implementing this protocol.

Example: Custom PostgreSQL Storage
----------------------------------
>>> from surfaces.custom_test_function.storage import Storage
>>> import psycopg2
>>>
>>> class PostgresStorage(Storage):
...     def __init__(self, connection_string: str, experiment: str):
...         self.conn = psycopg2.connect(connection_string)
...         self.experiment = experiment
...         self._setup_tables()
...
...     def save_evaluation(self, evaluation: dict) -> None:
...         # Insert into PostgreSQL
...         ...
...
...     def load_evaluations(self) -> List[dict]:
...         # Query from PostgreSQL
...         ...
...
...     # ... implement other methods
>>>
>>> # Use with CustomTestFunction
>>> func = CustomTestFunction(
...     objective_fn=my_func,
...     search_space={...},
...     storage=PostgresStorage("postgresql://...", "my-experiment"),
... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Storage(ABC):
    """Abstract base class for storage backends.

    All storage implementations must inherit from this class and implement
    the abstract methods. This ensures compatibility with CustomTestFunction.

    The storage is responsible for:
    - Persisting evaluation data (params, score, metadata)
    - Loading historical evaluations
    - Saving/loading function state (for checkpointing)
    - Managing experiment metadata

    Thread Safety
    -------------
    Implementations should be thread-safe if concurrent access is expected.
    The built-in SQLiteStorage uses connection-per-thread for safety.

    Methods to Implement
    --------------------
    - save_evaluation: Save a single evaluation record
    - load_evaluations: Load all evaluations for this experiment
    - save_state: Save complete function state (checkpoint)
    - load_state: Load function state from checkpoint
    - delete_experiment: Remove all data for this experiment
    - close: Clean up resources

    Optional Methods
    ----------------
    - query: Advanced querying (default raises NotImplementedError)
    - list_experiments: List all experiments (default raises NotImplementedError)
    """

    @property
    @abstractmethod
    def experiment(self) -> str:
        """The experiment name/identifier."""
        ...

    @abstractmethod
    def save_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """Save a single evaluation record.

        Parameters
        ----------
        evaluation : dict
            Evaluation data containing:
            - All parameter key-value pairs
            - "score": float - the objective value
            - "_timestamp": float - Unix timestamp (optional, added if missing)
            - "_evaluation_id": int - sequential ID (optional, added if missing)

        Example
        -------
        >>> storage.save_evaluation({
        ...     "x": 1.0,
        ...     "y": 2.0,
        ...     "score": 5.0,
        ...     "_timestamp": 1704067200.0,
        ...     "_evaluation_id": 1,
        ... })
        """
        ...

    @abstractmethod
    def load_evaluations(self) -> List[Dict[str, Any]]:
        """Load all evaluations for this experiment.

        Returns
        -------
        list of dict
            List of evaluation records, each containing parameters and score.
            Should be ordered by evaluation ID (chronological order).

        Example
        -------
        >>> evaluations = storage.load_evaluations()
        >>> print(evaluations[0])
        {"x": 1.0, "y": 2.0, "score": 5.0, "_timestamp": 1704067200.0}
        """
        ...

    @abstractmethod
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save complete function state for checkpointing.

        Parameters
        ----------
        state : dict
            Complete state including:
            - "n_evaluations": int
            - "best_score": float or None
            - "best_params": dict or None
            - "total_time": float
            - "memory_cache": dict (parameter tuples -> scores)
            - "metadata": dict (user metadata)
            - "_checkpoint_timestamp": float

        Notes
        -----
        This overwrites any previous state for this experiment.
        """
        ...

    @abstractmethod
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load function state from checkpoint.

        Returns
        -------
        dict or None
            The saved state, or None if no checkpoint exists.
        """
        ...

    @abstractmethod
    def delete_experiment(self) -> None:
        """Delete all data for this experiment.

        This removes all evaluations, state, and metadata.
        Use with caution - this operation cannot be undone.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the storage and release resources.

        Called when the CustomTestFunction is deleted or explicitly closed.
        Should handle being called multiple times gracefully.
        """
        ...

    # =========================================================================
    # Optional Methods (with default implementations)
    # =========================================================================

    def query(
        self,
        filter_fn: Optional[callable] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query evaluations with filtering and ordering.

        This is an optional advanced feature. The default implementation
        loads all evaluations and filters in Python.

        Parameters
        ----------
        filter_fn : callable, optional
            Function that takes an evaluation dict and returns True to include.
        order_by : str, optional
            Field name to order by. Prefix with "-" for descending.
        limit : int, optional
            Maximum number of results to return.

        Returns
        -------
        list of dict
            Filtered and ordered evaluations.

        Example
        -------
        >>> # Get top 10 evaluations
        >>> best = storage.query(order_by="score", limit=10)
        >>>
        >>> # Get evaluations where x > 0
        >>> positive_x = storage.query(filter_fn=lambda e: e["x"] > 0)
        """
        evaluations = self.load_evaluations()

        if filter_fn is not None:
            evaluations = [e for e in evaluations if filter_fn(e)]

        if order_by is not None:
            descending = order_by.startswith("-")
            key = order_by.lstrip("-")
            evaluations = sorted(
                evaluations,
                key=lambda e: e.get(key, 0),
                reverse=descending,
            )

        if limit is not None:
            evaluations = evaluations[:limit]

        return evaluations

    def list_experiments(self) -> List[str]:
        """List all experiments in this storage.

        Returns
        -------
        list of str
            Experiment names/identifiers.

        Raises
        ------
        NotImplementedError
            If the storage backend doesn't support listing experiments.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support listing experiments")

    def experiment_exists(self) -> bool:
        """Check if this experiment exists in storage.

        Returns
        -------
        bool
            True if the experiment has any data.
        """
        try:
            evaluations = self.load_evaluations()
            return len(evaluations) > 0
        except Exception:
            return False

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def __enter__(self) -> "Storage":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing storage."""
        self.close()
