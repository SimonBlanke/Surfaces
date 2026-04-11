# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Persistent manifest tracking surrogate model training status.

The manifest is a single JSON file living next to the ONNX models in
the models/ directory. It records which functions have been trained,
their quality metrics, and whether the config has changed since last
training (staleness detection via config hashing).

The manifest is updated atomically after each function completes, so
a crash mid-run preserves the state of all previously finished functions.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_MANIFEST_PATH = MODELS_DIR / "manifest.json"


def compute_config_hash(
    config: Dict[str, Any],
    max_samples: Optional[int] = None,
    hidden_layers: Optional[tuple] = None,
    max_iter: Optional[int] = None,
) -> str:
    """Compute a stable hash of all parameters that affect model quality.

    Includes the registry config (search space structure) and the
    training parameters. A model trained with --max-samples 5 produces
    a different hash than a full run, so the full run will correctly
    detect it as stale.
    """
    hashable = {
        "fixed_params": config["fixed_params"],
        "hyperparams": config["hyperparams"],
        "fidelity_levels": config.get("fidelity_levels", [1.0]),
        "max_samples": max_samples,
        "hidden_layers": list(hidden_layers) if hidden_layers else None,
        "max_iter": max_iter,
    }
    raw = json.dumps(hashable, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


class SurrogateManifest:
    """Read/write the surrogate training manifest.

    Parameters
    ----------
    path : Path, optional
        Path to manifest.json. Defaults to models/manifest.json.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_MANIFEST_PATH
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"version": 1, "functions": {}}

    def _save(self):
        """Write manifest atomically via tmp-file + rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
            f.write("\n")
        os.replace(str(tmp), str(self.path))

    def get(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get manifest entry for a function, or None if absent."""
        return self._data["functions"].get(function_name)

    def record_success(
        self,
        function_name: str,
        config_hash: str,
        metrics: Dict[str, float],
        n_errors: int,
        n_timeouts: int,
        fidelity_aware: bool,
        duration: float,
        onnx_path: str,
    ):
        """Record a successful training run and persist immediately."""
        self._data["functions"][function_name] = {
            "status": "ok",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "duration": round(duration, 1),
            "config_hash": config_hash,
            "metrics": {
                "r2": round(metrics["r2"], 6),
                "mse": round(metrics["mse"], 8),
                "n_samples": int(metrics["n_samples"]),
            },
            "n_errors": n_errors,
            "n_timeouts": n_timeouts,
            "fidelity_aware": fidelity_aware,
            "onnx_path": onnx_path,
            "error": None,
        }
        self._data["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def record_failure(
        self,
        function_name: str,
        config_hash: str,
        error: Exception,
        duration: float,
    ):
        """Record a failed training run and persist immediately."""
        import traceback

        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        tb_short = "".join(tb_lines[-3:])

        self._data["functions"][function_name] = {
            "status": "failed",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "duration": round(duration, 1),
            "config_hash": config_hash,
            "error": {
                "type": type(error).__name__,
                "message": str(error)[:500],
                "traceback": tb_short[:2000],
            },
        }
        self._data["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def needs_training(self, function_name: str, config_hash: str) -> bool:
        """Check whether a function needs (re)training.

        Returns True when: never trained, previously failed, or the
        registry config changed since the last successful training.
        """
        entry = self.get(function_name)
        if entry is None:
            return True
        if entry["status"] != "ok":
            return True
        if entry.get("config_hash") != config_hash:
            return True
        return False

    def get_reason(self, function_name: str, config_hash: str) -> str:
        """Human-readable reason why a function needs training."""
        entry = self.get(function_name)
        if entry is None:
            return "never trained"
        if entry["status"] == "failed":
            err = entry.get("error", {})
            return f"previously failed ({err.get('type', '?')}: {err.get('message', '?')[:60]})"
        if entry.get("config_hash") != config_hash:
            return "config changed (stale)"
        return "up to date"

    def failed_functions(self) -> List[str]:
        """Return names of functions with status=failed."""
        return [
            name for name, entry in self._data["functions"].items() if entry["status"] == "failed"
        ]

    def summary(self) -> Dict[str, int]:
        """Counts by status."""
        funcs = self._data["functions"]
        counts = {"total": len(funcs), "ok": 0, "failed": 0, "stale": 0}
        for entry in funcs.values():
            status = entry.get("status", "unknown")
            if status in counts:
                counts[status] += 1
        return counts

    def format_status_table(self, registry_configs: Optional[Dict[str, Dict]] = None) -> str:
        """Format a human-readable status table.

        Parameters
        ----------
        registry_configs : dict, optional
            Mapping of function_name -> config dict from the registry.
            When provided, staleness is checked against current config hashes.
        """
        lines = []
        lines.append(
            f"{'Function':<40s} {'Status':<10s} {'R2':>8s} {'Samples':>8s} {'Time':>10s} {'Trained':>20s}"
        )
        lines.append("-" * 100)

        for name, entry in sorted(self._data["functions"].items()):
            status = entry.get("status", "?")

            if registry_configs and status == "ok":
                cfg = registry_configs.get(name)
                if cfg and entry.get("config_hash") != compute_config_hash(cfg):
                    status = "stale"

            if status == "ok":
                m = entry.get("metrics", {})
                r2 = f"{m.get('r2', 0):.4f}"
                samples = str(m.get("n_samples", "?"))
                dur = _format_duration(entry.get("duration", 0))
            elif status == "failed":
                err = entry.get("error", {})
                r2 = "FAILED"
                samples = err.get("type", "?")
                dur = _format_duration(entry.get("duration", 0))
            else:
                r2 = status
                samples = ""
                dur = ""

            trained = entry.get("trained_at", "")[:19].replace("T", " ")
            lines.append(
                f"{name:<40s} {status:<10s} {r2:>8s} {samples:>8s} {dur:>10s} {trained:>20s}"
            )

        s = self.summary()
        lines.append("-" * 100)
        lines.append(f"Total: {s['total']}  ok: {s['ok']}  failed: {s['failed']}")
        return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"
