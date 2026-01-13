# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""In-memory storage implementation.

This is the default storage backend that keeps all data in memory.
Data is lost when the Python process exits.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ._protocol import Storage


class InMemoryStorage(Storage):
    """In-memory storage backend (no persistence).

    This is the default storage used when no storage is specified.
    All data is kept in Python lists/dicts and lost when the process exits.

    Use this for:
    - Quick experiments where persistence isn't needed
    - Testing and development
    - When you want maximum performance (no I/O overhead)

    Parameters
    ----------
    experiment : str, default="default"
        Experiment name/identifier.

    Examples
    --------
    >>> storage = InMemoryStorage(experiment="my-test")
    >>> storage.save_evaluation({"x": 1.0, "score": 1.0})
    >>> storage.load_evaluations()
    [{"x": 1.0, "score": 1.0, "_timestamp": ..., "_evaluation_id": 1}]
    """

    def __init__(self, experiment: str = "default") -> None:
        self._experiment = experiment
        self._evaluations: List[Dict[str, Any]] = []
        self._state: Optional[Dict[str, Any]] = None
        self._next_id: int = 1

    @property
    def experiment(self) -> str:
        """The experiment name/identifier."""
        return self._experiment

    def save_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """Save an evaluation to memory."""
        record = evaluation.copy()

        # Add metadata if not present
        if "_timestamp" not in record:
            record["_timestamp"] = time.time()
        if "_evaluation_id" not in record:
            record["_evaluation_id"] = self._next_id
            self._next_id += 1

        self._evaluations.append(record)

    def load_evaluations(self) -> List[Dict[str, Any]]:
        """Load all evaluations from memory."""
        return self._evaluations.copy()

    def save_state(self, state: Dict[str, Any]) -> None:
        """Save state to memory."""
        self._state = state.copy()
        self._state["_checkpoint_timestamp"] = time.time()

    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load state from memory."""
        if self._state is None:
            return None
        return self._state.copy()

    def delete_experiment(self) -> None:
        """Clear all data."""
        self._evaluations = []
        self._state = None
        self._next_id = 1

    def close(self) -> None:
        """No-op for in-memory storage."""
        pass

    def __repr__(self) -> str:
        return (
            f"InMemoryStorage(experiment='{self._experiment}', "
            f"n_evaluations={len(self._evaluations)})"
        )
