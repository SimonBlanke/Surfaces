# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Surrogate trainer for ML functions with fixed parameters.

This module provides tools for training surrogates across all
(hyperparameter, dataset, cv) combinations for ML functions.

Developer Usage:
    from surfaces._surrogates import (
        train_ml_surrogate,
        train_all_ml_surrogates,
        train_missing_ml_surrogates,
        list_ml_surrogates,
    )

    # Train one specific surrogate
    train_ml_surrogate("k_neighbors_classifier")

    # Train all registered surrogates
    train_all_ml_surrogates()

    # Train only missing surrogates
    train_missing_ml_surrogates()
"""

import json
import signal
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._error_aggregator import ErrorAggregator
from ._ml_registry import (
    get_function_config,
    get_registered_functions,
)
from ._surrogate_loader import get_surrogate_path

# Suppress sklearn warnings during training
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class _EvalTimeout(Exception):
    """Raised when a single evaluation exceeds its time budget."""


def _timeout_handler(signum, frame):
    raise _EvalTimeout()


# Output directory for models
MODELS_DIR = Path(__file__).parent / "models"


def _serialize_value(v):
    """Convert numpy scalars and other non-JSON types to Python natives."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


class MLSurrogateTrainer:
    """Train surrogate for an ML function across all fixed param combinations.

    Parameters
    ----------
    function_name : str
        Name of the registered ML function (e.g., "k_neighbors_classifier").
    output_dir : Path, optional
        Directory to save models. Defaults to _surrogates/models/.
    verbose : bool, default=True
        Print progress information.
    logger : logging.Logger, optional
        Logger instance for structured logging. When provided, messages
        go to the logger instead of stdout.

    Examples
    --------
    >>> trainer = MLSurrogateTrainer("k_neighbors_classifier")
    >>> trainer.collect_data()
    >>> trainer.train()
    >>> trainer.export()
    """

    def __init__(
        self,
        function_name: str,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
        logger=None,
    ):
        self.function_name = function_name
        self.config = get_function_config(function_name)
        self.output_dir = output_dir or MODELS_DIR
        self.verbose = verbose
        self._logger = logger

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.param_names: List[str] = []
        self.param_encodings: Dict[str, Dict[str, int]] = {}
        self.model = None
        self.metrics: Dict[str, float] = {}
        self.n_errors: int = 0

    def _log(self, msg: str, level: str = "info"):
        """Route message to logger or stdout."""
        if self._logger:
            getattr(self._logger, level)(msg)
        elif self.verbose:
            print(msg)

    def collect_data(
        self,
        max_samples_per_combo: Optional[int] = None,
        progress_callback: Optional[callable] = None,
        eval_timeout: Optional[int] = 60,
        evaluation_store=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect training data across all (HP, fixed_param, fidelity) combinations.

        When fidelity_levels contains more than just [1.0], the trainer
        evaluates each HP combination at every fidelity level. The resulting
        surrogate can then predict scores for arbitrary fidelity values.

        When an EvaluationStore is provided, existing evaluations are
        loaded from the database and only missing combinations are
        evaluated. New results are persisted back to the store.

        Parameters
        ----------
        max_samples_per_combo : int, optional
            Max samples per fixed parameter combination.
        progress_callback : callable, optional
            Called as ``progress_callback(done, total)`` after each
            evaluation. Allows callers to display a progress bar or
            log periodic updates.
        eval_timeout : int, optional
            Maximum seconds per single evaluation. Evaluations exceeding
            this limit are skipped and logged as timeouts. Set to None to
            disable. Default: 60 seconds.
        evaluation_store : EvaluationStore, optional
            Persistent store for caching evaluations across runs. When
            provided, only missing combinations are evaluated.

        Returns
        -------
        X, y : tuple of ndarray
            Training data.
        """
        FuncClass = self.config["class"]
        fixed_params = self.config["fixed_params"]
        hyperparams = self.config["hyperparams"]
        fidelity_levels = self.config.get("fidelity_levels", [1.0])

        self.fidelity_aware = len(fidelity_levels) > 1 or fidelity_levels != [1.0]

        fixed_keys = list(fixed_params.keys())
        fixed_values = [fixed_params[k] for k in fixed_keys]
        fixed_combos = list(product(*fixed_values))

        self._log(f"Collecting data for {self.function_name}")
        self._log(f"  Fixed params: {fixed_keys} ({len(fixed_combos)} combinations)")
        if self.fidelity_aware:
            self._log(f"  Fidelity levels: {fidelity_levels}")
        if eval_timeout:
            self._log(f"  Eval timeout: {eval_timeout}s")

        # Check for cached evaluations in the store
        existing_keys = set()
        function_hash = None
        if evaluation_store is not None:
            from ._evaluation_store import compute_function_hash

            function_hash = compute_function_hash(FuncClass)
            existing_keys = evaluation_store.get_existing_keys(self.function_name, function_hash)
            if existing_keys:
                self._log(f"  Found {len(existing_keys)} cached evaluations in store")

        # Pre-compute total evaluation count for progress reporting.
        first_func = FuncClass(**dict(zip(fixed_keys, fixed_combos[0])), use_surrogate=False)
        n_hp = len(list(product(*[first_func.search_space[k] for k in first_func.search_space])))
        if max_samples_per_combo and n_hp > max_samples_per_combo:
            n_hp = max_samples_per_combo
        total_evals = len(fixed_combos) * n_hp * len(fidelity_levels)
        self._log(f"  Estimated evaluations: {total_evals}")

        use_alarm = eval_timeout is not None and hasattr(signal, "SIGALRM")
        if eval_timeout and not use_alarm:
            self._log(
                "  SIGALRM not available (Windows?), timeout disabled",
                level="warning",
            )

        new_records = []
        new_db_records = []
        error_agg = ErrorAggregator(logger=self._logger)
        n_timeouts = 0
        n_nan = 0
        n_cached = 0
        done = 0
        start_time = time.time()

        for fixed_combo in fixed_combos:
            fixed_dict = dict(zip(fixed_keys, fixed_combo))
            func = FuncClass(**fixed_dict, use_surrogate=False)
            search_space = func.search_space

            hp_keys = list(search_space.keys())
            hp_values = [search_space[k] for k in hp_keys]
            hp_combos = list(product(*hp_values))

            if max_samples_per_combo and len(hp_combos) > max_samples_per_combo:
                indices = np.random.choice(len(hp_combos), max_samples_per_combo, replace=False)
                hp_combos = [hp_combos[i] for i in indices]

            for hp_combo in hp_combos:
                hp_dict = dict(zip(hp_keys, hp_combo))

                for fidelity in fidelity_levels:
                    # Skip if we already have this evaluation in the store
                    if existing_keys:
                        params_json = json.dumps(
                            {k: _serialize_value(v) for k, v in hp_dict.items()},
                            sort_keys=True,
                        )
                        if (params_json, fidelity) in existing_keys:
                            n_cached += 1
                            done += 1
                            if progress_callback is not None:
                                progress_callback(done, total_evals)
                            continue

                    t_eval = time.time()
                    try:
                        if use_alarm:
                            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                            signal.alarm(eval_timeout)

                        func._active_fidelity = fidelity if fidelity < 1.0 else None
                        score = func._objective(hp_dict)

                        if use_alarm:
                            signal.alarm(0)

                        eval_time = time.time() - t_eval

                        if np.isnan(score):
                            n_nan += 1
                        else:
                            record = {**hp_dict, **fixed_dict, "_score": score}
                            if self.fidelity_aware:
                                record["__fidelity__"] = fidelity
                            new_records.append(record)

                            if evaluation_store is not None:
                                params_json = json.dumps(
                                    {k: _serialize_value(v) for k, v in hp_dict.items()},
                                    sort_keys=True,
                                )
                                new_db_records.append(
                                    {
                                        "params_json": params_json,
                                        "fidelity": fidelity,
                                        "score": score,
                                        "dataset": fixed_dict.get("dataset"),
                                        "cv": fixed_dict.get("cv"),
                                        "eval_time_s": eval_time,
                                    }
                                )
                    except _EvalTimeout:
                        n_timeouts += 1
                        self._log(
                            f"  Timeout ({eval_timeout}s): {fixed_dict} | "
                            f"{hp_dict} | fidelity={fidelity}",
                            level="warning",
                        )
                        if evaluation_store is not None:
                            params_json = json.dumps(
                                {k: _serialize_value(v) for k, v in hp_dict.items()},
                                sort_keys=True,
                            )
                            new_db_records.append(
                                {
                                    "params_json": params_json,
                                    "fidelity": fidelity,
                                    "score": None,
                                    "error_type": "Timeout",
                                    "dataset": fixed_dict.get("dataset"),
                                    "cv": fixed_dict.get("cv"),
                                    "eval_time_s": time.time() - t_eval,
                                }
                            )
                    except Exception as exc:
                        context = f"{fixed_dict} | {hp_dict} | fidelity={fidelity}"
                        error_agg.record(exc, context)
                        if evaluation_store is not None:
                            params_json = json.dumps(
                                {k: _serialize_value(v) for k, v in hp_dict.items()},
                                sort_keys=True,
                            )
                            new_db_records.append(
                                {
                                    "params_json": params_json,
                                    "fidelity": fidelity,
                                    "score": None,
                                    "error_type": type(exc).__name__,
                                    "dataset": fixed_dict.get("dataset"),
                                    "cv": fixed_dict.get("cv"),
                                    "eval_time_s": time.time() - t_eval,
                                }
                            )
                    finally:
                        if use_alarm:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old_handler)
                        func._active_fidelity = None

                    done += 1
                    if progress_callback is not None:
                        progress_callback(done, total_evals)

            # Batch-insert after each fixed_param combo to limit memory
            # and persist progress in case of a crash
            if evaluation_store is not None and new_db_records:
                evaluation_store.store_batch(self.function_name, function_hash, new_db_records)
                new_db_records = []

        elapsed = time.time() - start_time
        self.n_errors = error_agg.total
        self.n_timeouts = n_timeouts

        if n_cached > 0:
            self._log(f"  Skipped {n_cached} cached evaluations")

        self._log(
            f"  Collected {len(new_records)} new samples in {elapsed:.1f}s "
            f"({error_agg.total} errors, {n_timeouts} timeouts, {n_nan} NaN)"
        )
        if error_agg.total > 0:
            error_agg.log_summary()
        if n_timeouts > 0:
            self._log(
                f"  {n_timeouts} evaluations exceeded {eval_timeout}s timeout",
                level="warning",
            )

        # When using the store, load ALL evaluations (cached + new)
        # so the training data is complete
        if evaluation_store is not None and existing_keys:
            all_db_records = evaluation_store.load_all(self.function_name, function_hash)
            self._log(
                f"  Total data from store: {len(all_db_records)} evaluations "
                f"({n_cached} cached + {len(new_records)} new)"
            )
            records = self._convert_db_records(all_db_records, fixed_keys)
        else:
            records = new_records

        self._build_arrays(records, hyperparams, fixed_keys)
        return self.X, self.y

    def _convert_db_records(self, db_records: List[Dict], fixed_keys: List[str]) -> List[Dict]:
        """Convert DB records back to the internal record format."""
        records = []
        for r in db_records:
            record = dict(r)
            if self.fidelity_aware:
                record["__fidelity__"] = record.pop("fidelity", 1.0)
            else:
                record.pop("fidelity", None)
            records.append(record)
        return records

    def _build_arrays(self, records: List[Dict], hyperparams: List[str], fixed_keys: List[str]):
        """Convert records to numpy arrays with encodings."""
        # Determine param order: hyperparams first, then fixed params, then fidelity
        self.param_names = hyperparams + fixed_keys
        if self.fidelity_aware:
            self.param_names = self.param_names + ["__fidelity__"]

        # Build encodings for categorical params
        self.param_encodings = {}
        for key in self.param_names:
            values = [r[key] for r in records]
            unique_values = sorted(set(str(v) for v in values))

            # Check if all numeric
            try:
                [float(v) for v in unique_values]
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False

            if not is_numeric:
                self.param_encodings[key] = {v: i for i, v in enumerate(unique_values)}

        # Build X and y arrays
        X_list = []
        y_list = []

        for r in records:
            row = []
            for key in self.param_names:
                val = r[key]
                if key in self.param_encodings:
                    row.append(float(self.param_encodings[key][str(val)]))
                else:
                    row.append(float(val))
            X_list.append(row)
            y_list.append(r["_score"])

        self.X = np.array(X_list, dtype=np.float32)
        self.y = np.array(y_list, dtype=np.float32)

    def train(self, hidden_layers: Tuple[int, ...] = (64, 64), max_iter: int = 1000):
        """Train MLP surrogate model.

        Parameters
        ----------
        hidden_layers : tuple
            Hidden layer sizes for MLP.
        max_iter : int
            Maximum training iterations.
        """
        if self.X is None:
            raise ValueError("No data collected. Call collect_data() first.")

        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self._log("Training surrogate model...")
        start_time = time.time()

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layers,
                        max_iter=max_iter,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=42,
                        verbose=False,
                    ),
                ),
            ]
        )

        pipeline.fit(self.X, self.y)
        self.model = pipeline

        # Evaluate
        y_pred = pipeline.predict(self.X)
        mse = float(np.mean((self.y - y_pred) ** 2))
        r2 = float(1 - mse / np.var(self.y))

        self.metrics = {
            "mse": mse,
            "r2": r2,
            "training_time": time.time() - start_time,
            "n_samples": len(self.y),
        }

        self._log(f"  R2: {r2:.4f}, MSE: {mse:.6f}")

    def export(self) -> Path:
        """Export trained model to ONNX format.

        Returns
        -------
        Path
            Path to exported ONNX file.
        """
        if self.model is None:
            raise ValueError("No model trained. Call train() first.")

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError("skl2onnx required. Install: pip install skl2onnx")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{self.function_name}.onnx"
        metadata_path = self.output_dir / f"{self.function_name}.onnx.meta.json"

        # Convert to ONNX
        n_features = self.X.shape[1]
        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(self.model, initial_types=initial_type)

        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # Save metadata
        metadata = {
            "function_name": self.function_name,
            "param_names": self.param_names,
            "param_encodings": self.param_encodings,
            "fidelity_aware": getattr(self, "fidelity_aware", False),
            "fidelity_levels": self.config.get("fidelity_levels", [1.0]),
            "n_samples": int(self.metrics["n_samples"]),
            "n_invalid_samples": self.n_errors,
            "has_validity_model": False,
            "y_range": [float(self.y.min()), float(self.y.max())],
            "training_mse": self.metrics["mse"],
            "training_r2": self.metrics["r2"],
            "training_time": self.metrics["training_time"],
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self._log(f"Exported: {output_path}")
        return output_path


def train_ml_surrogate(
    function_name: str,
    verbose: bool = True,
    max_samples_per_combo: Optional[int] = None,
) -> Path:
    """Train surrogate for a single ML function.

    Parameters
    ----------
    function_name : str
        Name of registered function (e.g., "k_neighbors_classifier").
    verbose : bool
        Print progress.
    max_samples_per_combo : int, optional
        Limit samples per fixed param combination.

    Returns
    -------
    Path
        Path to exported ONNX file.
    """
    trainer = MLSurrogateTrainer(function_name, verbose=verbose)
    trainer.collect_data(max_samples_per_combo)
    trainer.train()
    return trainer.export()


def train_all_ml_surrogates(verbose: bool = True) -> List[Path]:
    """Train surrogates for all registered ML functions.

    Parameters
    ----------
    verbose : bool
        Print progress.

    Returns
    -------
    list of Path
        Paths to exported ONNX files.
    """
    paths = []
    functions = get_registered_functions()

    if verbose:
        print(f"Training {len(functions)} ML surrogates...")
        print("=" * 50)

    for name in functions:
        if verbose:
            print(f"\n[{name}]")
        path = train_ml_surrogate(name, verbose=verbose)
        paths.append(path)

    if verbose:
        print("\n" + "=" * 50)
        print(f"Done! Trained {len(paths)} surrogates.")

    return paths


def train_missing_ml_surrogates(verbose: bool = True) -> List[Path]:
    """Train surrogates only for functions without existing models.

    Parameters
    ----------
    verbose : bool
        Print progress.

    Returns
    -------
    list of Path
        Paths to newly exported ONNX files.
    """
    paths = []
    functions = get_registered_functions()
    missing = [f for f in functions if get_surrogate_path(f) is None]

    if not missing:
        if verbose:
            print("All surrogates exist. Nothing to train.")
        return paths

    if verbose:
        print(f"Training {len(missing)} missing surrogates...")
        print(f"  Missing: {missing}")
        print("=" * 50)

    for name in missing:
        if verbose:
            print(f"\n[{name}]")
        path = train_ml_surrogate(name, verbose=verbose)
        paths.append(path)

    if verbose:
        print("\n" + "=" * 50)
        print(f"Done! Trained {len(paths)} surrogates.")

    return paths


def list_ml_surrogates() -> Dict[str, Dict]:
    """List all registered ML functions and their surrogate status.

    Returns
    -------
    dict
        Function name -> {"exists": bool, "path": Path or None}
    """
    result = {}
    for name in get_registered_functions():
        path = get_surrogate_path(name)
        result[name] = {
            "exists": path is not None,
            "path": path,
        }
    return result
