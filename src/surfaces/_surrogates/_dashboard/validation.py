# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Validation orchestration for the dashboard.

Performs surrogate validation with database tracking.
Note: We implement validation directly here rather than using SurrogateValidator
because the validator doesn't properly handle fixed params for ML functions.
"""

import time
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .database import (
    get_all_surrogates,
    insert_validation_run,
)


def _sample_random(search_space: Dict, n_samples: int, seed: int) -> List[Dict]:
    """Generate random parameter samples."""
    np.random.seed(seed)
    samples = []
    for _ in range(n_samples):
        params = {}
        for name, values in search_space.items():
            params[name] = np.random.choice(values)
        samples.append(params)
    return samples


def _sample_grid(search_space: Dict, max_samples: Optional[int] = None) -> List[Dict]:
    """Generate grid parameter samples."""
    param_names = sorted(search_space.keys())
    grid_points = list(product(*[search_space[name] for name in param_names]))

    if max_samples and len(grid_points) > max_samples:
        indices = np.random.choice(len(grid_points), max_samples, replace=False)
        grid_points = [grid_points[i] for i in indices]

    samples = []
    for point in grid_points:
        params = {name: val for name, val in zip(param_names, point)}
        samples.append(params)
    return samples


def validate_surrogate(
    function_name: str,
    validation_type: str = "random",
    n_samples: int = 100,
    random_seed: int = 42,
    progress_callback: Optional[Callable[[str], None]] = None,
    db_path: Optional[Path] = None,
) -> dict:
    """Validate a surrogate against the real function.

    Parameters
    ----------
    function_name : str
        Name of the function to validate.
    validation_type : str
        'random' or 'grid'.
    n_samples : int
        Number of samples for validation.
    random_seed : int
        Random seed for reproducibility.
    progress_callback : callable, optional
        Function to call with progress messages.
    db_path : Path, optional
        Path to SQLite database.

    Returns
    -------
    dict
        Result with keys: success, message, run_id, metrics, timing, data
    """
    from .._ml_registry import get_function_config
    from .._surrogate_loader import load_surrogate

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    log(f"Starting {validation_type} validation for {function_name} ({n_samples} samples)")

    try:
        # Get function class and create instance
        config = get_function_config(function_name)
        FuncClass = config["class"]

        # Use default fixed params
        fixed_params = config["fixed_params"]
        default_fixed = {k: v[0] for k, v in fixed_params.items()}

        log(f"Creating function instance with: {default_fixed}")

        # Create real function instance
        real_func = FuncClass(**default_fixed, use_surrogate=False)

        # Load surrogate directly
        surrogate = load_surrogate(function_name)
        if surrogate is None:
            raise ValueError(f"No surrogate model found for {function_name}")

        # Get search space and generate samples
        search_space = real_func.search_space
        if validation_type == "random":
            samples = _sample_random(search_space, n_samples, random_seed)
        else:
            samples = _sample_grid(search_space, n_samples)

        log(f"Running validation on {len(samples)} samples...")

        # Run validation
        y_real = []
        y_surr = []
        errors = []
        real_times = []
        surr_times = []

        for i, params in enumerate(samples):
            try:
                # Real evaluation
                start = time.time()
                real = real_func.pure_objective_function(params)
                real_time = time.time() - start

                if np.isnan(real):
                    continue

                # Surrogate evaluation - need to merge with fixed params
                surrogate_params = {**params, **default_fixed}
                start = time.time()
                surr = surrogate.predict(surrogate_params)
                surr_time = time.time() - start

                y_real.append(real)
                y_surr.append(surr)
                errors.append(real - surr)
                real_times.append(real_time)
                surr_times.append(surr_time)

                if (i + 1) % 20 == 0:
                    log(f"  Evaluated {i + 1}/{len(samples)} samples")

            except Exception:
                pass  # Skip invalid combinations

        if len(y_real) == 0:
            raise ValueError("No valid samples were evaluated")

        y_real = np.array(y_real)
        y_surr = np.array(y_surr)
        errors = np.array(errors)

        # Compute metrics
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        max_error = float(np.max(np.abs(errors)))

        # R2 score
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        # Correlation
        correlation = float(np.corrcoef(y_real, y_surr)[0, 1])

        # Timing
        avg_real_time = float(np.mean(real_times) * 1000)  # ms
        avg_surr_time = float(np.mean(surr_times) * 1000)  # ms
        speedup = avg_real_time / avg_surr_time if avg_surr_time > 0 else 0

        metrics = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "max_error": max_error,
            "correlation": correlation,
        }

        timing = {
            "avg_real_ms": avg_real_time,
            "avg_surrogate_ms": avg_surr_time,
            "speedup": speedup,
        }

        log(f"Validation complete. R2: {r2:.4f}, MAE: {mae:.4f}")
        log(f"Speedup: {speedup:.0f}x")

        # Store in database
        run_id = insert_validation_run(
            function_name=function_name,
            validation_type=validation_type,
            n_samples=len(y_real),
            metrics=metrics,
            timing=timing,
            random_seed=random_seed if validation_type == "random" else None,
            db_path=db_path,
        )

        log(f"Stored validation run {run_id}")

        return {
            "success": True,
            "message": f"Successfully validated {function_name}",
            "run_id": run_id,
            "metrics": metrics,
            "timing": timing,
            "data": {
                "y_real": y_real,
                "y_surrogate": y_surr,
                "errors": errors,
            },
        }

    except Exception as e:
        error_msg = str(e)
        log(f"Validation failed: {error_msg}")
        return {
            "success": False,
            "message": f"Validation failed: {error_msg}",
            "run_id": None,
            "metrics": None,
            "timing": None,
            "data": None,
        }


def validate_all(
    validation_type: str = "random",
    n_samples: int = 100,
    random_seed: int = 42,
    progress_callback: Optional[Callable[[str], None]] = None,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """Validate all surrogates that exist.

    Parameters
    ----------
    validation_type : str
        'random' or 'grid'.
    n_samples : int
        Number of samples for validation.
    random_seed : int
        Random seed for reproducibility.
    progress_callback : callable, optional
        Function to call with progress messages.
    db_path : Path, optional
        Path to SQLite database.

    Returns
    -------
    list of dict
        Results for each validation attempt.
    """

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Get functions with surrogates
    surrogates = get_all_surrogates(db_path)
    with_surrogate = [s["function_name"] for s in surrogates if s["has_surrogate"]]

    if not with_surrogate:
        log("No surrogates to validate.")
        return []

    log(f"Validating {len(with_surrogate)} surrogates...")

    results = []
    for i, name in enumerate(with_surrogate):
        log(f"\n[{i+1}/{len(with_surrogate)}] Validating {name}...")
        result = validate_surrogate(
            name, validation_type, n_samples, random_seed, progress_callback, db_path
        )
        results.append(result)

    return results
