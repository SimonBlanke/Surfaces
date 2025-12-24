# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Synchronization between .meta.json files and SQLite database.

The .meta.json files are the source of truth for model metadata.
This module syncs that data to SQLite for dashboard queries.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

from .._onnx_utils import get_metadata_path, get_surrogate_model_path
from .database import init_db, upsert_surrogate


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    if file_path is None or not file_path.exists():
        return ""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_meta_json(meta_path: Path) -> dict:
    """Load and parse a .meta.json file."""
    with open(meta_path, "r") as f:
        return json.load(f)


def get_function_type(function_name: str) -> str:
    """Determine if function is classification or regression."""
    if "classifier" in function_name:
        return "classification"
    elif "regressor" in function_name:
        return "regression"
    return "unknown"


def sync_all(db_path: Optional[Path] = None) -> dict:
    """Sync all ML functions from registry to database.

    This function:
    1. Gets all registered ML functions from the registry
    2. Checks if each has a .onnx and .meta.json file (in data package or local)
    3. Upserts the data into SQLite

    Parameters
    ----------
    db_path : Path, optional
        Path to SQLite database.

    Returns
    -------
    dict
        Sync statistics.
    """
    # Initialize database
    init_db(db_path)

    # Import registry (lazy to avoid circular imports)
    from .._ml_registry import get_registered_functions

    functions = get_registered_functions()
    stats = {
        "total": len(functions),
        "with_surrogate": 0,
        "without_surrogate": 0,
        "synced": 0,
    }

    for function_name in functions:
        onnx_path = get_surrogate_model_path(function_name)
        meta_path = get_metadata_path(function_name)

        has_surrogate = onnx_path is not None and meta_path is not None

        if has_surrogate:
            stats["with_surrogate"] += 1
            metadata = load_meta_json(meta_path)
            file_hash = compute_file_hash(onnx_path)
        else:
            stats["without_surrogate"] += 1
            metadata = None
            file_hash = None

        function_type = get_function_type(function_name)

        upsert_surrogate(
            function_name=function_name,
            function_type=function_type,
            has_surrogate=has_surrogate,
            onnx_file_hash=file_hash,
            metadata=metadata,
            db_path=db_path,
        )
        stats["synced"] += 1

    return stats


def sync_single(
    function_name: str,
    db_path: Optional[Path] = None,
) -> bool:
    """Sync a single function to the database.

    Parameters
    ----------
    function_name : str
        Name of the function to sync.
    db_path : Path, optional
        Path to SQLite database.

    Returns
    -------
    bool
        True if surrogate exists, False otherwise.
    """
    onnx_path = get_surrogate_model_path(function_name)
    meta_path = get_metadata_path(function_name)

    has_surrogate = onnx_path is not None and meta_path is not None

    if has_surrogate:
        metadata = load_meta_json(meta_path)
        file_hash = compute_file_hash(onnx_path)
    else:
        metadata = None
        file_hash = None

    function_type = get_function_type(function_name)

    upsert_surrogate(
        function_name=function_name,
        function_type=function_type,
        has_surrogate=has_surrogate,
        onnx_file_hash=file_hash,
        metadata=metadata,
        db_path=db_path,
    )

    return has_surrogate


def check_sync_needed(
    function_name: str,
    db_path: Optional[Path] = None,
) -> bool:
    """Check if a function needs to be re-synced.

    Compares file hash in database with current file hash.

    Parameters
    ----------
    function_name : str
        Name of the function to check.
    db_path : Path, optional
        Path to SQLite database.

    Returns
    -------
    bool
        True if sync is needed, False otherwise.
    """
    from .database import get_surrogate

    onnx_path = get_surrogate_model_path(function_name)

    # Get current hash
    current_hash = compute_file_hash(onnx_path) if onnx_path is not None else None

    # Get stored hash
    surrogate = get_surrogate(function_name, db_path)
    if surrogate is None:
        return True  # Not in database, needs sync

    stored_hash = surrogate.get("onnx_file_hash")

    # Compare
    if current_hash != stored_hash:
        return True

    return False
