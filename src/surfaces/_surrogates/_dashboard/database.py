# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
SQLite database for tracking surrogate models and their metrics.

This module provides:
- Database schema creation and migrations
- CRUD operations for surrogates, validation runs, and training jobs
- Query helpers for dashboard views
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Database file location (alongside dashboard code)
DB_PATH = Path(__file__).parent / "surrogates.db"


# =============================================================================
# Schema
# =============================================================================

SCHEMA = """
-- Core surrogate info (synced from .meta.json)
CREATE TABLE IF NOT EXISTS surrogates (
    function_name TEXT PRIMARY KEY,
    function_type TEXT,               -- 'classification' or 'regression'

    -- From .meta.json
    param_names TEXT,                 -- JSON array
    param_encodings TEXT,             -- JSON object
    n_samples INTEGER,
    n_invalid_samples INTEGER,
    has_validity_model BOOLEAN,
    y_range_min REAL,
    y_range_max REAL,
    training_time_sec REAL,
    training_mse REAL,
    training_r2 REAL,

    -- Tracking fields
    onnx_file_hash TEXT,              -- SHA256 to detect changes
    has_surrogate BOOLEAN DEFAULT 0,  -- Whether .onnx file exists

    -- Timestamps
    last_synced_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Validation history (multiple runs per surrogate)
CREATE TABLE IF NOT EXISTS validation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    function_name TEXT REFERENCES surrogates(function_name),
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Config
    validation_type TEXT,             -- 'random' or 'grid'
    n_samples INTEGER,
    random_seed INTEGER,              -- For reproducibility

    -- Accuracy metrics
    r2_score REAL,
    mae REAL,
    rmse REAL,
    max_error REAL,
    correlation REAL,

    -- Performance metrics
    avg_real_time_ms REAL,
    avg_surrogate_time_ms REAL,
    speedup_factor REAL
);

-- Training job history
CREATE TABLE IF NOT EXISTS training_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    function_name TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT DEFAULT 'running',    -- 'running', 'completed', 'failed'
    error_message TEXT,
    triggered_by TEXT                 -- 'manual', 'missing', 'low_accuracy', 'retrain_all'
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_validation_function ON validation_runs(function_name);
CREATE INDEX IF NOT EXISTS idx_validation_date ON validation_runs(validated_at);
CREATE INDEX IF NOT EXISTS idx_training_function ON training_jobs(function_name);
CREATE INDEX IF NOT EXISTS idx_training_status ON training_jobs(status);
"""


# =============================================================================
# Connection Management
# =============================================================================


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize the database with schema."""
    path = db_path or DB_PATH
    with get_connection(path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Get a database connection as a context manager."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def dict_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert sqlite3.Row to dict."""
    return dict(zip(row.keys(), row))


# =============================================================================
# Surrogate CRUD
# =============================================================================


def upsert_surrogate(
    function_name: str,
    function_type: str,
    has_surrogate: bool,
    onnx_file_hash: Optional[str] = None,
    metadata: Optional[Dict] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Insert or update a surrogate record.

    Parameters
    ----------
    function_name : str
        Unique function identifier.
    function_type : str
        'classification' or 'regression'.
    has_surrogate : bool
        Whether .onnx file exists.
    onnx_file_hash : str, optional
        SHA256 hash of .onnx file.
    metadata : dict, optional
        Parsed .meta.json contents.
    """
    now = datetime.now().isoformat()
    meta = metadata or {}

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO surrogates (
                function_name, function_type, has_surrogate, onnx_file_hash,
                param_names, param_encodings, n_samples, n_invalid_samples,
                has_validity_model, y_range_min, y_range_max,
                training_time_sec, training_mse, training_r2,
                last_synced_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(function_name) DO UPDATE SET
                function_type = excluded.function_type,
                has_surrogate = excluded.has_surrogate,
                onnx_file_hash = excluded.onnx_file_hash,
                param_names = excluded.param_names,
                param_encodings = excluded.param_encodings,
                n_samples = excluded.n_samples,
                n_invalid_samples = excluded.n_invalid_samples,
                has_validity_model = excluded.has_validity_model,
                y_range_min = excluded.y_range_min,
                y_range_max = excluded.y_range_max,
                training_time_sec = excluded.training_time_sec,
                training_mse = excluded.training_mse,
                training_r2 = excluded.training_r2,
                last_synced_at = excluded.last_synced_at,
                updated_at = excluded.updated_at
            """,
            (
                function_name,
                function_type,
                has_surrogate,
                onnx_file_hash,
                json.dumps(meta.get("param_names", [])),
                json.dumps(meta.get("param_encodings", {})),
                meta.get("n_samples"),
                meta.get("n_invalid_samples"),
                meta.get("has_validity_model", False),
                meta.get("y_range", [None, None])[0],
                meta.get("y_range", [None, None])[1],
                meta.get("training_time"),
                meta.get("training_mse"),
                meta.get("training_r2"),
                now,
                now,
            ),
        )
        conn.commit()


def get_surrogate(function_name: str, db_path: Optional[Path] = None) -> Optional[Dict]:
    """Get a single surrogate by name."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM surrogates WHERE function_name = ?",
            (function_name,),
        ).fetchone()
        if row:
            result = dict_from_row(row)
            # Parse JSON fields
            result["param_names"] = json.loads(result["param_names"] or "[]")
            result["param_encodings"] = json.loads(result["param_encodings"] or "{}")
            return result
        return None


def get_all_surrogates(db_path: Optional[Path] = None) -> List[Dict]:
    """Get all surrogate records."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM surrogates ORDER BY function_name").fetchall()
        results = []
        for row in rows:
            result = dict_from_row(row)
            result["param_names"] = json.loads(result["param_names"] or "[]")
            result["param_encodings"] = json.loads(result["param_encodings"] or "{}")
            results.append(result)
        return results


# =============================================================================
# Validation Runs CRUD
# =============================================================================


def insert_validation_run(
    function_name: str,
    validation_type: str,
    n_samples: int,
    metrics: Dict[str, float],
    timing: Dict[str, float],
    random_seed: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Insert a validation run record.

    Returns
    -------
    int
        The ID of the inserted record.
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO validation_runs (
                function_name, validation_type, n_samples, random_seed,
                r2_score, mae, rmse, max_error, correlation,
                avg_real_time_ms, avg_surrogate_time_ms, speedup_factor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                function_name,
                validation_type,
                n_samples,
                random_seed,
                metrics.get("r2"),
                metrics.get("mae"),
                metrics.get("rmse"),
                metrics.get("max_error"),
                metrics.get("correlation"),
                timing.get("avg_real_ms"),
                timing.get("avg_surrogate_ms"),
                timing.get("speedup"),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_validation_runs(
    function_name: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> List[Dict]:
    """Get validation runs, optionally filtered by function."""
    with get_connection(db_path) as conn:
        if function_name:
            rows = conn.execute(
                """
                SELECT * FROM validation_runs
                WHERE function_name = ?
                ORDER BY validated_at DESC
                LIMIT ?
                """,
                (function_name, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM validation_runs
                ORDER BY validated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict_from_row(row) for row in rows]


def get_latest_validation(
    function_name: str,
    db_path: Optional[Path] = None,
) -> Optional[Dict]:
    """Get the most recent validation run for a function."""
    runs = get_validation_runs(function_name, limit=1, db_path=db_path)
    return runs[0] if runs else None


# =============================================================================
# Training Jobs CRUD
# =============================================================================


def insert_training_job(
    function_name: str,
    triggered_by: str = "manual",
    db_path: Optional[Path] = None,
) -> int:
    """Insert a training job record (status='running').

    Returns
    -------
    int
        The ID of the inserted record.
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO training_jobs (function_name, triggered_by, status)
            VALUES (?, ?, 'running')
            """,
            (function_name, triggered_by),
        )
        conn.commit()
        return cursor.lastrowid


def update_training_job(
    job_id: int,
    status: str,
    error_message: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Update a training job's status."""
    now = datetime.now().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET status = ?, completed_at = ?, error_message = ?
            WHERE id = ?
            """,
            (status, now, error_message, job_id),
        )
        conn.commit()


def get_training_jobs(
    function_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> List[Dict]:
    """Get training jobs with optional filters."""
    with get_connection(db_path) as conn:
        query = "SELECT * FROM training_jobs WHERE 1=1"
        params = []

        if function_name:
            query += " AND function_name = ?"
            params.append(function_name)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict_from_row(row) for row in rows]


# =============================================================================
# Dashboard Queries
# =============================================================================


def get_overview_data(db_path: Optional[Path] = None) -> List[Dict]:
    """Get overview data for all functions with latest validation metrics.

    Returns a list of dicts with surrogate info plus latest R2 score.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                s.*,
                v.r2_score as latest_r2,
                v.validated_at as latest_validation_at
            FROM surrogates s
            LEFT JOIN (
                SELECT function_name, r2_score, validated_at,
                       ROW_NUMBER() OVER (PARTITION BY function_name ORDER BY validated_at DESC) as rn
                FROM validation_runs
            ) v ON s.function_name = v.function_name AND v.rn = 1
            ORDER BY s.function_name
            """
        ).fetchall()

        results = []
        for row in rows:
            result = dict_from_row(row)
            result["param_names"] = json.loads(result["param_names"] or "[]")
            result["param_encodings"] = json.loads(result["param_encodings"] or "{}")
            results.append(result)
        return results


def get_functions_needing_training(
    r2_threshold: float = 0.95,
    db_path: Optional[Path] = None,
) -> List[str]:
    """Get function names that need training (missing or low accuracy)."""
    with get_connection(db_path) as conn:
        # Functions without surrogates
        missing = conn.execute(
            "SELECT function_name FROM surrogates WHERE has_surrogate = 0"
        ).fetchall()

        # Functions with low R2
        low_accuracy = conn.execute(
            """
            SELECT s.function_name
            FROM surrogates s
            LEFT JOIN (
                SELECT function_name, r2_score,
                       ROW_NUMBER() OVER (PARTITION BY function_name ORDER BY validated_at DESC) as rn
                FROM validation_runs
            ) v ON s.function_name = v.function_name AND v.rn = 1
            WHERE s.has_surrogate = 1
              AND (v.r2_score IS NULL OR v.r2_score < ?)
            """,
            (r2_threshold,),
        ).fetchall()

        return list(set([r[0] for r in missing] + [r[0] for r in low_accuracy]))


def get_dashboard_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get aggregate statistics for the dashboard."""
    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM surrogates").fetchone()[0]
        with_surrogate = conn.execute(
            "SELECT COUNT(*) FROM surrogates WHERE has_surrogate = 1"
        ).fetchone()[0]
        total_validations = conn.execute("SELECT COUNT(*) FROM validation_runs").fetchone()[0]
        total_trainings = conn.execute(
            "SELECT COUNT(*) FROM training_jobs WHERE status = 'completed'"
        ).fetchone()[0]

        return {
            "total_functions": total,
            "with_surrogate": with_surrogate,
            "without_surrogate": total - with_surrogate,
            "total_validations": total_validations,
            "total_trainings": total_trainings,
        }
