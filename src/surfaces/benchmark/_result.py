"""Benchmark result storage and analysis."""

from __future__ import annotations

from typing import Any

from surfaces._benchmark._trace import Trace


class BenchmarkResult:
    """Results from a benchmark run.

    Stores traces indexed by (function_name, optimizer_name, seed)
    and provides aggregation, filtering, and export methods.
    """

    def __init__(self, traces: dict[tuple[str, str, int], Trace]):
        self._traces = dict(traces)

    @property
    def function_names(self) -> list[str]:
        """Sorted list of unique function names in this result."""
        return sorted({k[0] for k in self._traces})

    @property
    def optimizer_names(self) -> list[str]:
        """Sorted list of unique optimizer names in this result."""
        return sorted({k[1] for k in self._traces})

    @property
    def seeds(self) -> list[int]:
        """Sorted list of unique seeds in this result."""
        return sorted({k[2] for k in self._traces})

    @property
    def n_traces(self) -> int:
        return len(self._traces)

    def traces(
        self,
        function: str | None = None,
        optimizer: str | None = None,
        seed: int | None = None,
    ) -> dict[tuple[str, str, int], Trace]:
        """Filter traces by function name, optimizer name, and/or seed."""
        result = {}
        for (f, o, s), trace in self._traces.items():
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
        """Generate a summary table of benchmark results.

        Parameters
        ----------
        at_cu : float, optional
            Report best score at this CU budget.
        at_iter : int, optional
            Report best score at this iteration count.
        """
        lines = []

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
        header += f"  {'Evals':>{col_w['evals']}}" f"  {'Overhead%':>{col_w['oh']}}"
        lines.append(header)
        lines.append("-" * len(header))

        for func_name in self.function_names:
            for opt_name in self.optimizer_names:
                matching = [
                    t for (f, o, _s), t in self._traces.items() if f == func_name and o == opt_name
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

                row += (
                    f"  {mean_evals:>{col_w['evals']}.0f}"
                    f"  {overhead_pct:>{col_w['oh'] - 1}.1f}%"
                )
                lines.append(row)

        return "\n".join(lines)

    def to_dataframe(self):
        """Export all evaluation records as a pandas DataFrame.

        Each row is a single evaluation with function/optimizer/seed
        metadata and full cost breakdown.
        """
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for (func_name, opt_name, seed), trace in self._traces.items():
            for i, record in enumerate(trace):
                row = {
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
        n_funcs = len(self.function_names)
        n_opts = len(self.optimizer_names)
        n_seeds = len(self.seeds)
        return (
            f"BenchmarkResult("
            f"{n_funcs} functions, "
            f"{n_opts} optimizers, "
            f"{n_seeds} seeds, "
            f"{self.n_traces} traces)"
        )
