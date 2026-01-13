# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""SQLite storage implementation using Python stdlib.

This provides persistent storage using SQLite3 with no external dependencies.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ._protocol import Storage


class SQLiteStorage(Storage):
    """SQLite-based persistent storage.

    Uses Python's built-in sqlite3 module for zero-dependency persistence.
    Data is stored in a local SQLite database file.

    Thread Safety
    -------------
    This implementation uses connection-per-thread to ensure thread safety.
    Each thread gets its own database connection.

    Parameters
    ----------
    path : str or Path
        Path to the SQLite database file. Will be created if it doesn't exist.
        Use ":memory:" for an in-memory SQLite database (useful for testing).
    experiment : str
        Experiment name/identifier. Used to separate different experiments
        in the same database file.

    Examples
    --------
    Basic usage:

    >>> storage = SQLiteStorage("./experiments.db", experiment="my-experiment")
    >>> storage.save_evaluation({"x": 1.0, "y": 2.0, "score": 5.0})
    >>> storage.load_evaluations()
    [{"x": 1.0, "y": 2.0, "score": 5.0, ...}]

    Auto-create directory:

    >>> storage = SQLiteStorage("./data/experiments.db", experiment="test")
    # Creates ./data/ directory if it doesn't exist

    Multiple experiments in one database:

    >>> storage1 = SQLiteStorage("./exp.db", experiment="exp-v1")
    >>> storage2 = SQLiteStorage("./exp.db", experiment="exp-v2")
    # Both use the same file but different tables
    """

    def __init__(
        self,
        path: Union[str, Path],
        experiment: str,
    ) -> None:
        self._path = Path(path) if path != ":memory:" else path
        self._experiment = experiment
        self._local = threading.local()

        # Create directory if needed
        if isinstance(self._path, Path) and self._path.parent:
            self._path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize tables
        self._setup_tables()

    @property
    def experiment(self) -> str:
        """The experiment name/identifier."""
        return self._experiment

    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            path = str(self._path) if isinstance(self._path, Path) else self._path
            self._local.conn = sqlite3.connect(path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _setup_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self._conn.cursor()

        # Evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment TEXT NOT NULL,
                evaluation_id INTEGER NOT NULL,
                params_json TEXT NOT NULL,
                score REAL NOT NULL,
                timestamp REAL NOT NULL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Index for fast experiment lookup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_evaluations_experiment
            ON evaluations(experiment)
        """)

        # State/checkpoint table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment TEXT UNIQUE NOT NULL,
                state_json TEXT NOT NULL,
                timestamp REAL NOT NULL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Experiments metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment TEXT PRIMARY KEY,
                metadata_json TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        self._conn.commit()

    def save_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """Save an evaluation to the database."""
        # Separate score and metadata from params
        record = evaluation.copy()
        score = record.pop("score")
        timestamp = record.pop("_timestamp", time.time())
        evaluation_id = record.pop("_evaluation_id", None)

        # Get next evaluation_id if not provided
        if evaluation_id is None:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT MAX(evaluation_id) FROM evaluations WHERE experiment = ?",
                (self._experiment,),
            )
            result = cursor.fetchone()[0]
            evaluation_id = (result or 0) + 1

        # Store remaining fields as params
        params_json = json.dumps(record)

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO evaluations (experiment, evaluation_id, params_json, score, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (self._experiment, evaluation_id, params_json, score, timestamp),
        )
        self._conn.commit()

    def load_evaluations(self) -> List[Dict[str, Any]]:
        """Load all evaluations from the database."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT params_json, score, timestamp, evaluation_id
            FROM evaluations
            WHERE experiment = ?
            ORDER BY evaluation_id ASC
            """,
            (self._experiment,),
        )

        evaluations = []
        for row in cursor.fetchall():
            record = json.loads(row["params_json"])
            record["score"] = row["score"]
            record["_timestamp"] = row["timestamp"]
            record["_evaluation_id"] = row["evaluation_id"]
            evaluations.append(record)

        return evaluations

    def save_state(self, state: Dict[str, Any]) -> None:
        """Save checkpoint state to the database."""
        state_copy = state.copy()
        state_copy["_checkpoint_timestamp"] = time.time()

        # Convert memory_cache keys from tuples to strings for JSON
        if "memory_cache" in state_copy:
            cache = state_copy["memory_cache"]
            state_copy["memory_cache"] = {str(k): v for k, v in cache.items()}

        state_json = json.dumps(state_copy)

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO checkpoints (experiment, state_json, timestamp)
            VALUES (?, ?, ?)
            """,
            (self._experiment, state_json, time.time()),
        )
        self._conn.commit()

    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint state from the database."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT state_json FROM checkpoints WHERE experiment = ?",
            (self._experiment,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        state = json.loads(row["state_json"])

        # Convert memory_cache keys back to tuples
        if "memory_cache" in state:
            cache = state["memory_cache"]
            state["memory_cache"] = {eval(k): v for k, v in cache.items()}

        return state

    def delete_experiment(self) -> None:
        """Delete all data for this experiment."""
        cursor = self._conn.cursor()
        cursor.execute(
            "DELETE FROM evaluations WHERE experiment = ?",
            (self._experiment,),
        )
        cursor.execute(
            "DELETE FROM checkpoints WHERE experiment = ?",
            (self._experiment,),
        )
        cursor.execute(
            "DELETE FROM experiments WHERE experiment = ?",
            (self._experiment,),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def list_experiments(self) -> List[str]:
        """List all experiments in this database."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT DISTINCT experiment FROM evaluations")
        return [row[0] for row in cursor.fetchall()]

    def query(
        self,
        filter_fn: Optional[callable] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query evaluations with optional SQL optimization."""
        # Build SQL query
        sql = """
            SELECT params_json, score, timestamp, evaluation_id
            FROM evaluations
            WHERE experiment = ?
        """
        params = [self._experiment]

        # Add ORDER BY if it's a known column
        if order_by is not None:
            descending = order_by.startswith("-")
            key = order_by.lstrip("-")
            if key == "score":
                direction = "DESC" if descending else "ASC"
                sql += f" ORDER BY score {direction}"
            elif key == "_timestamp":
                direction = "DESC" if descending else "ASC"
                sql += f" ORDER BY timestamp {direction}"
            else:
                # Fall back to Python sorting for param fields
                sql += " ORDER BY evaluation_id ASC"

        if limit is not None and filter_fn is None:
            sql += f" LIMIT {int(limit)}"

        cursor = self._conn.cursor()
        cursor.execute(sql, params)

        evaluations = []
        for row in cursor.fetchall():
            record = json.loads(row["params_json"])
            record["score"] = row["score"]
            record["_timestamp"] = row["timestamp"]
            record["_evaluation_id"] = row["evaluation_id"]
            evaluations.append(record)

        # Apply Python filter if provided
        if filter_fn is not None:
            evaluations = [e for e in evaluations if filter_fn(e)]
            if limit is not None:
                evaluations = evaluations[:limit]

        # Apply Python sorting for non-SQL columns
        if order_by is not None and order_by.lstrip("-") not in ("score", "_timestamp"):
            descending = order_by.startswith("-")
            key = order_by.lstrip("-")
            evaluations = sorted(
                evaluations,
                key=lambda e: e.get(key, 0),
                reverse=descending,
            )

        return evaluations

    def __repr__(self) -> str:
        n_evals = len(self.load_evaluations())
        return (
            f"SQLiteStorage(path='{self._path}', "
            f"experiment='{self._experiment}', "
            f"n_evaluations={n_evals})"
        )
