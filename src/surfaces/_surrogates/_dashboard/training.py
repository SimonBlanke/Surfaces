# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Training orchestration for the dashboard.

Wraps the MLSurrogateTrainer with database tracking.
"""

from pathlib import Path
from typing import Callable, List, Optional

from .database import (
    get_all_surrogates,
    get_functions_needing_training,
    insert_training_job,
    update_training_job,
)
from .sync import sync_single


def train_surrogate(
    function_name: str,
    triggered_by: str = "manual",
    progress_callback: Optional[Callable[[str], None]] = None,
    db_path: Optional[Path] = None,
) -> dict:
    """Train a surrogate for a single function with database tracking.

    Parameters
    ----------
    function_name : str
        Name of the function to train.
    triggered_by : str
        What triggered this training ('manual', 'missing', 'low_accuracy', 'retrain_all').
    progress_callback : callable, optional
        Function to call with progress messages.
    db_path : Path, optional
        Path to SQLite database.

    Returns
    -------
    dict
        Result with keys: success, message, job_id, metrics
    """
    from .._ml_surrogate_trainer import MLSurrogateTrainer

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Create job record
    job_id = insert_training_job(function_name, triggered_by, db_path)
    log(f"Started training job {job_id} for {function_name}")

    try:
        # Train
        trainer = MLSurrogateTrainer(function_name, verbose=False)

        log("Collecting training data...")
        trainer.collect_data()

        log(f"Collected {len(trainer.y)} samples")
        log("Training model...")
        trainer.train()

        log(f"Training complete. R2: {trainer.metrics['r2']:.4f}")
        log("Exporting ONNX model...")
        trainer.export()

        # Update job status
        update_training_job(job_id, "completed", db_path=db_path)

        # Re-sync to database
        log("Syncing to database...")
        sync_single(function_name, db_path=db_path)

        return {
            "success": True,
            "message": f"Successfully trained {function_name}",
            "job_id": job_id,
            "metrics": trainer.metrics,
        }

    except Exception as e:
        error_msg = str(e)
        update_training_job(job_id, "failed", error_msg, db_path)
        log(f"Training failed: {error_msg}")
        return {
            "success": False,
            "message": f"Training failed: {error_msg}",
            "job_id": job_id,
            "metrics": None,
        }


def train_missing(
    progress_callback: Optional[Callable[[str], None]] = None,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """Train surrogates for functions without existing models.

    Parameters
    ----------
    progress_callback : callable, optional
        Function to call with progress messages.
    db_path : Path, optional
        Path to SQLite database.

    Returns
    -------
    list of dict
        Results for each training attempt.
    """

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Get functions without surrogates
    surrogates = get_all_surrogates(db_path)
    missing = [s["function_name"] for s in surrogates if not s["has_surrogate"]]

    if not missing:
        log("All functions have surrogates. Nothing to train.")
        return []

    log(f"Found {len(missing)} functions without surrogates: {missing}")

    results = []
    for i, name in enumerate(missing):
        log(f"\n[{i+1}/{len(missing)}] Training {name}...")
        result = train_surrogate(name, "missing", progress_callback, db_path)
        results.append(result)

    return results


def train_low_accuracy(
    r2_threshold: float = 0.95,
    progress_callback: Optional[Callable[[str], None]] = None,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """Train surrogates with validation R2 below threshold.

    Parameters
    ----------
    r2_threshold : float
        R2 threshold below which to retrain.
    progress_callback : callable, optional
        Function to call with progress messages.
    db_path : Path, optional
        Path to SQLite database.

    Returns
    -------
    list of dict
        Results for each training attempt.
    """

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Get functions needing training
    needing = get_functions_needing_training(r2_threshold, db_path)

    # Filter to only those that have surrogates (low accuracy, not missing)
    surrogates = get_all_surrogates(db_path)
    has_surrogate = {s["function_name"] for s in surrogates if s["has_surrogate"]}
    low_accuracy = [n for n in needing if n in has_surrogate]

    if not low_accuracy:
        log(f"No surrogates with R2 < {r2_threshold}. Nothing to retrain.")
        return []

    log(f"Found {len(low_accuracy)} surrogates with R2 < {r2_threshold}: {low_accuracy}")

    results = []
    for i, name in enumerate(low_accuracy):
        log(f"\n[{i+1}/{len(low_accuracy)}] Retraining {name}...")
        result = train_surrogate(name, "low_accuracy", progress_callback, db_path)
        results.append(result)

    return results


def train_all(
    progress_callback: Optional[Callable[[str], None]] = None,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """Train/retrain all registered surrogates.

    Parameters
    ----------
    progress_callback : callable, optional
        Function to call with progress messages.
    db_path : Path, optional
        Path to SQLite database.

    Returns
    -------
    list of dict
        Results for each training attempt.
    """

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    surrogates = get_all_surrogates(db_path)
    all_functions = [s["function_name"] for s in surrogates]

    log(f"Training all {len(all_functions)} surrogates...")

    results = []
    for i, name in enumerate(all_functions):
        log(f"\n[{i+1}/{len(all_functions)}] Training {name}...")
        result = train_surrogate(name, "retrain_all", progress_callback, db_path)
        results.append(result)

    return results
