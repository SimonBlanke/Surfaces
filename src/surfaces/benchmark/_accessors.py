"""Accessor classes that organize Benchmark functionality into namespaces.

ResultAccessor  - querying and analyzing results
IOAccessor      - saving and loading benchmark state
PlotAccessor    - visualization (planned)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from surfaces.benchmark._trace import Trace

if TYPE_CHECKING:
    from surfaces.benchmark._benchmark import Benchmark


class _NumpyEncoder(json.JSONEncoder):
    """Handles numpy scalars and arrays in JSON serialization."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ResultAccessor:
    """Query and analyze benchmark results.

    Accessed via ``bench.results``. All methods operate on the
    accumulated traces inside the parent Benchmark instance.
    """

    def __init__(self, benchmark: Benchmark) -> None:
        self._bench = benchmark

    @property
    def function_names(self) -> list[str]:
        """Sorted unique function names across all traces."""
        return sorted({k[0] for k in self._bench._traces})

    @property
    def optimizer_names(self) -> list[str]:
        """Sorted unique optimizer names across all traces."""
        return sorted({k[1] for k in self._bench._traces})

    @property
    def seeds(self) -> list[int]:
        """Sorted unique seeds across all traces."""
        return sorted({k[2] for k in self._bench._traces})

    @property
    def n_traces(self) -> int:
        """Total number of traces."""
        return len(self._bench._traces)

    def traces(
        self,
        function: str | None = None,
        optimizer: str | None = None,
        seed: int | None = None,
    ) -> dict[tuple[str, str, int], Trace]:
        """Filter traces by function name, optimizer name, and/or seed."""
        result = {}
        for (f, o, s), trace in self._bench._traces.items():
            if function is not None and f != function:
                continue
            if optimizer is not None and o != optimizer:
                continue
            if seed is not None and s != seed:
                continue
            result[(f, o, s)] = trace
        return result

    def summary(
        self,
        at_cu: float | None = None,
        at_iter: int | None = None,
    ) -> str:
        """Generate a formatted summary table of benchmark results.

        Parameters
        ----------
        at_cu : float, optional
            Report best score at this CU budget.
        at_iter : int, optional
            Report best score at this iteration count.
        """
        traces = self._bench._traces
        if not traces:
            return "No traces recorded yet. Call run() first."

        func_names = self.function_names
        opt_names = self.optimizer_names

        col_w = {"func": 30, "opt": 25, "seeds": 5, "best": 12, "evals": 8, "oh": 10}

        header = (
            f"{'Function':<{col_w['func']}} "
            f"{'Optimizer':<{col_w['opt']}} "
            f"{'Seeds':>{col_w['seeds']}} "
            f"{'Best':>{col_w['best']}}"
        )
        if at_cu is not None:
            header += f"  {'@CU=' + str(int(at_cu)):>15}"
        if at_iter is not None:
            header += f"  {'@iter=' + str(at_iter):>12}"
        header += f"  {'Evals':>{col_w['evals']}}  {'Overhead%':>{col_w['oh']}}"

        lines = [header, "-" * len(header)]

        for func_name in func_names:
            for opt_name in opt_names:
                matching = [
                    t for (f, o, _s), t in traces.items() if f == func_name and o == opt_name
                ]
                if not matching:
                    continue

                n_seeds = len(matching)
                best_scores = [t.best_score for t in matching if t.best_score is not None]
                if best_scores:
                    mean_best = sum(best_scores) / len(best_scores)
                    best_str = f"{mean_best:.4f}"
                else:
                    best_str = "n/a"

                mean_evals = sum(t.n_evaluations for t in matching) / n_seeds
                total_cu = sum(t.total_cu for t in matching)
                total_overhead = sum(t.total_overhead_cu for t in matching)
                overhead_pct = total_overhead / total_cu * 100 if total_cu > 0 else 0

                row = (
                    f"{func_name:<{col_w['func']}} "
                    f"{opt_name:<{col_w['opt']}} "
                    f"{n_seeds:>{col_w['seeds']}} "
                    f"{best_str:>{col_w['best']}}"
                )

                if at_cu is not None:
                    scores = [t.score_at_cu(at_cu) for t in matching]
                    valid = [s for s in scores if s is not None]
                    if valid:
                        val = f"{sum(valid) / len(valid):.4f}"
                    else:
                        val = "n/a"
                    row += f"  {val:>15}"

                if at_iter is not None:
                    scores = [t.score_at_iter(at_iter) for t in matching]
                    valid = [s for s in scores if s is not None]
                    if valid:
                        val = f"{sum(valid) / len(valid):.4f}"
                    else:
                        val = "n/a"
                    row += f"  {val:>12}"

                row += f"  {mean_evals:>{col_w['evals']}.0f}  {overhead_pct:>{col_w['oh'] - 1}.1f}%"
                lines.append(row)

        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Export all evaluation records as a pandas DataFrame.

        Each row is a single evaluation with function/optimizer/seed
        metadata and full cost breakdown. Parameter values are expanded
        into individual columns.
        """
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for (func_name, opt_name, seed), trace in self._bench._traces.items():
            for i, record in enumerate(trace):
                row: dict[str, Any] = {
                    "function": func_name,
                    "optimizer": opt_name,
                    "seed": seed,
                    "iteration": i + 1,
                    "score": record.score,
                    "best_so_far": record.best_so_far,
                    "eval_cu": record.eval_cu,
                    "overhead_cu": record.overhead_cu,
                    "cumulative_cu": record.cumulative_cu,
                    "wall_seconds": record.wall_seconds,
                }
                row.update(record.params)
                rows.append(row)

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"ResultAccessor({self.n_traces} traces, "
            f"{len(self.function_names)} functions, "
            f"{len(self.optimizer_names)} optimizers)"
        )


class IOAccessor:
    """Save benchmark state (config + results) to disk.

    Accessed via ``bench.io``. Loading is done via the
    ``Benchmark.load()`` classmethod.
    """

    def __init__(self, benchmark: Benchmark) -> None:
        self._bench = benchmark

    def save(self, path: str | Path) -> None:
        """Save the full benchmark state to a JSON file.

        Stores configuration (budget, seeds), registered functions
        and optimizers, the Surfaces version, and all accumulated
        traces. The file can be loaded back with ``Benchmark.load()``.

        Parameters
        ----------
        path : str or Path
            Output file path. Will be overwritten if it exists.
        """
        import surfaces

        path = Path(path)

        functions = [f"{cls.__module__}.{cls.__qualname__}" for cls in self._bench._functions]

        optimizers = []
        for obj, params in self._bench._optimizers:
            if isinstance(obj, type):
                class_path = f"{obj.__module__}.{obj.__qualname__}"
            else:
                class_path = f"{type(obj).__module__}.{type(obj).__qualname__}"
            optimizers.append({"class": class_path, "params": params})

        trace_list = []
        for (func_name, opt_name, seed), trace in sorted(self._bench._traces.items()):
            trace_list.append(
                {
                    "function": func_name,
                    "optimizer": opt_name,
                    "seed": seed,
                    "records": [
                        {
                            "params": record.params,
                            "score": record.score,
                            "eval_cu": record.eval_cu,
                            "overhead_cu": record.overhead_cu,
                            "cumulative_cu": record.cumulative_cu,
                            "best_so_far": record.best_so_far,
                            "wall_seconds": record.wall_seconds,
                        }
                        for record in trace
                    ],
                }
            )

        data = {
            "format_version": 2,
            "surfaces_version": surfaces.__version__,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "budget_cu": self._bench._budget_cu,
                "budget_iter": self._bench._budget_iter,
                "n_seeds": self._bench._n_seeds,
                "seed": self._bench._seed,
            },
            "functions": functions,
            "optimizers": optimizers,
            "traces": trace_list,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)

    def __repr__(self) -> str:
        return "IOAccessor(use .save(path) to persist, Benchmark.load(path) to restore)"


class PlotAccessor:
    """Benchmark visualization (planned for a future release).

    Accessed via ``bench.plot``.
    """

    def __init__(self, benchmark: Benchmark) -> None:
        self._bench = benchmark

    def __repr__(self) -> str:
        return "PlotAccessor(not yet implemented)"
