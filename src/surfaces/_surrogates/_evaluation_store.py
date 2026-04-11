# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""SQLite-backed persistent store for ML function evaluations.

Separates data collection from model training. Evaluations are stored
once and reused across runs, so only missing combinations need to be
evaluated when the search space changes. A function hash (based on
source code and dependency versions) detects when stored results are
no longer valid.

The DB file lives next to the ONNX models in the models/ directory.
"""

import hashlib
import importlib.metadata
import inspect
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type

MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_DB_PATH = MODELS_DIR / "evaluations.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    function_name   TEXT    NOT NULL,
    function_hash   TEXT    NOT NULL,
    dataset         TEXT,
    cv              INTEGER,
    params_json     TEXT    NOT NULL,
    fidelity        REAL    NOT NULL,
    score           REAL,
    error_type      TEXT,
    eval_time_s     REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_eval_lookup
    ON evaluations (function_name, function_hash);
"""


def compute_function_hash(func_class: Type) -> str:
    """Hash a function class based on its source and dependency versions.

    The hash changes when:
    - The _ml_objective (or _objective) source code changes
    - A declared dependency changes its major.minor version

    Patch versions are ignored since they rarely affect numerical
    results and would cause unnecessary re-evaluations.
    """
    parts = []

    for method_name in ("_ml_objective", "_objective"):
        method = getattr(func_class, method_name, None)
        if method is not None:
            try:
                parts.append(inspect.getsource(method))
            except (OSError, TypeError):
                pass

    dep_versions = {}
    for dep in getattr(func_class, "_dependencies", ()):
        try:
            version = importlib.metadata.version(dep)
            major_minor = ".".join(version.split(".")[:2])
            dep_versions[dep] = major_minor
        except importlib.metadata.PackageNotFoundError:
            dep_versions[dep] = "not_installed"
    parts.append(json.dumps(dep_versions, sort_keys=True))

    raw = "\n".join(parts).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


class EvaluationStore:
    """Persistent SQLite store for function evaluation results.

    Parameters
    ----------
    db_path : Path, optional
        Path to the SQLite database. Defaults to models/evaluations.db.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), timeout=30)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self):
        self.conn.executescript(_CREATE_TABLE + _CREATE_INDEX)
        self.conn.commit()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def get_existing_keys(
        self,
        function_name: str,
        function_hash: str,
    ) -> Set[Tuple[str, float]]:
        """Return set of (params_json, fidelity) already evaluated.

        Only returns successful evaluations (score IS NOT NULL) for
        the given function and hash. Used to determine which
        combinations still need evaluation.
        """
        cursor = self.conn.execute(
            """
            SELECT params_json, fidelity FROM evaluations
            WHERE function_name = ? AND function_hash = ?
              AND score IS NOT NULL
            """,
            (function_name, function_hash),
        )
        return {(row["params_json"], row["fidelity"]) for row in cursor}

    def store_batch(
        self,
        function_name: str,
        function_hash: str,
        records: List[Dict[str, Any]],
    ):
        """Batch-insert evaluation records.

        Each record should have: params_json, fidelity, score (or None),
        and optionally dataset, cv, error_type, eval_time_s.
        """
        if not records:
            return

        self.conn.executemany(
            """
            INSERT INTO evaluations
                (function_name, function_hash, dataset, cv,
                 params_json, fidelity, score, error_type, eval_time_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    function_name,
                    function_hash,
                    r.get("dataset"),
                    r.get("cv"),
                    r["params_json"],
                    r["fidelity"],
                    r.get("score"),
                    r.get("error_type"),
                    r.get("eval_time_s"),
                )
                for r in records
            ],
        )
        self.conn.commit()

    def load_all(
        self,
        function_name: str,
        function_hash: str,
    ) -> List[Dict[str, Any]]:
        """Load all successful evaluations for a function+hash.

        Returns list of dicts with params (unpacked from JSON),
        fidelity, and score.
        """
        cursor = self.conn.execute(
            """
            SELECT params_json, fidelity, score, dataset, cv
            FROM evaluations
            WHERE function_name = ? AND function_hash = ?
              AND score IS NOT NULL
            """,
            (function_name, function_hash),
        )

        results = []
        for row in cursor:
            params = json.loads(row["params_json"])
            record = {**params, "_score": row["score"]}
            if row["dataset"] is not None:
                record["dataset"] = row["dataset"]
            if row["cv"] is not None:
                record["cv"] = row["cv"]
            record["fidelity"] = row["fidelity"]
            results.append(record)
        return results

    def count(self, function_name: str, function_hash: str) -> int:
        """Count successful evaluations for a function+hash."""
        cursor = self.conn.execute(
            """
            SELECT COUNT(*) FROM evaluations
            WHERE function_name = ? AND function_hash = ?
              AND score IS NOT NULL
            """,
            (function_name, function_hash),
        )
        return cursor.fetchone()[0]

    def invalidate(self, function_name: str):
        """Delete all evaluations for a function (all hashes)."""
        self.conn.execute(
            "DELETE FROM evaluations WHERE function_name = ?",
            (function_name,),
        )
        self.conn.commit()

    def invalidate_hash(self, function_name: str, function_hash: str):
        """Delete evaluations for a specific function+hash combination."""
        self.conn.execute(
            "DELETE FROM evaluations WHERE function_name = ? AND function_hash = ?",
            (function_name, function_hash),
        )
        self.conn.commit()

    def stats(self) -> Dict[str, Dict[str, Any]]:
        """Per-function statistics for inspection."""
        cursor = self.conn.execute(
            """
            SELECT function_name, function_hash,
                   COUNT(*) as total,
                   SUM(CASE WHEN score IS NOT NULL THEN 1 ELSE 0 END) as successful,
                   SUM(CASE WHEN error_type IS NOT NULL THEN 1 ELSE 0 END) as errors
            FROM evaluations
            GROUP BY function_name, function_hash
            ORDER BY function_name
            """
        )
        result = {}
        for row in cursor:
            name = row["function_name"]
            if name not in result:
                result[name] = []
            result[name].append(
                {
                    "hash": row["function_hash"],
                    "total": row["total"],
                    "successful": row["successful"],
                    "errors": row["errors"],
                }
            )
        return result

    def format_stats(self) -> str:
        """Human-readable stats table."""
        lines = []
        lines.append(f"{'Function':<40s} {'Hash':<18s} {'OK':>8s} {'Errors':>8s} {'Total':>8s}")
        lines.append("-" * 86)

        for name, entries in sorted(self.stats().items()):
            for entry in entries:
                lines.append(
                    f"{name:<40s} {entry['hash']:<18s} "
                    f"{entry['successful']:>8d} {entry['errors']:>8d} {entry['total']:>8d}"
                )
        return "\n".join(lines)
