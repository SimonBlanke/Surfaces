"""Accessor classes that organize Benchmark functionality into namespaces.

ResultAccessor  - querying and analyzing results
IOAccessor      - saving and loading benchmark state
PlotAccessor    - visualization (ECDF, convergence plots)
"""

from __future__ import annotations

import json
import warnings
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
        show_ci: bool = False,
    ) -> str:
        """Generate a formatted summary table of benchmark results.

        Parameters
        ----------
        at_cu : float, optional
            Report best score at this CU budget.
        at_iter : int, optional
            Report best score at this iteration count.
        show_ci : bool, default=False
            Show standard deviation and 95% confidence interval
            of the best score across seeds. Useful with >= 3 seeds.
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
        if show_ci:
            header += f"  {'Std':>8}  {'95% CI':>17}"
        if at_cu is not None:
            header += f"  {'@CU=' + str(int(at_cu)):>15}"
        if at_iter is not None:
            header += f"  {'@iter=' + str(at_iter):>12}"
        header += f"  {'Evals':>{col_w['evals']}}  {'Overhead%':>{col_w['oh']}}"

        lines = [header, "\u2500" * len(header)]

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
                    mean_best = None
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

                if show_ci:
                    if mean_best is not None and len(best_scores) >= 2:
                        variance = sum((s - mean_best) ** 2 for s in best_scores) / (
                            len(best_scores) - 1
                        )
                        std = variance**0.5
                        margin = 1.96 * std / len(best_scores) ** 0.5
                        row += f"  {std:>8.4f}  [{mean_best - margin:>7.3f},{mean_best + margin:>7.3f}]"
                    else:
                        row += f"  {'n/a':>8}  {'n/a':>17}"

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
                row.update({f"param_{k}": v for k, v in record.params.items()})
                rows.append(row)

        return pd.DataFrame(rows)

    def ert(
        self,
        precision: float = 1.0,
        targets: dict[str, float] | None = None,
    ) -> Any:
        """Compute Expected Running Time for all (function, optimizer) pairs.

        A problem counts as "solved" when ``best_so_far <= f_global + precision``.
        ERT follows the COCO convention: total budget across all seeds
        divided by the number of successful seeds.

        Parameters
        ----------
        precision : float, default=1.0
            Absolute distance from the known optimum (f_global).
        targets : dict, optional
            Per-function target scores. Overrides precision for
            functions present in this dict.

        Returns
        -------
        ERTTable
            Printable, subscriptable, and exportable to DataFrame.
        """
        from surfaces.benchmark._statistics import compute_ert

        optimal_scores = {cls.__name__: cls.f_global for cls in self._bench._functions}

        return compute_ert(
            self._bench._traces,
            optimal_scores,
            precision,
            targets,
        )

    def ranking(
        self,
        at_cu: float | None = None,
        alpha: float = 0.05,
        correction: str | None = "holm",
    ) -> Any:
        """Rank optimizers by normalized performance with pairwise tests.

        Scores are normalized per function (0 = worst observed,
        1 = best observed) and averaged over seeds. Ranks use
        tied-rank averaging within each function. Pairwise Wilcoxon
        signed-rank tests assess statistical significance.

        Parameters
        ----------
        at_cu : float, optional
            Evaluate scores at this CU budget instead of using the
            final best score.
        alpha : float, default=0.05
            Significance level for the Wilcoxon tests.
        correction : str or None, default="holm"
            Multiple comparison correction. ``"holm"`` applies the
            Holm step-down procedure (controls family-wise error rate).
            ``None`` returns raw uncorrected p-values.

        Returns
        -------
        RankingTable
            Printable, subscriptable, and exportable to DataFrame.
        """
        from surfaces.benchmark._statistics import compute_ranking

        return compute_ranking(self._bench._traces, alpha, at_cu, correction)

    def friedman(
        self,
        at_cu: float | None = None,
        alpha: float = 0.05,
    ) -> Any:
        """Friedman omnibus test for comparing multiple optimizers.

        Tests whether at least one optimizer's performance differs
        significantly. This is the recommended first step before
        pairwise comparisons: if the Friedman test does not reject,
        pairwise differences are not statistically supported.

        Requires at least 3 optimizers and 3 functions where all
        optimizers produced results.

        Parameters
        ----------
        at_cu : float, optional
            Evaluate scores at this CU budget.
        alpha : float, default=0.05
            Significance level.

        Returns
        -------
        FriedmanResult
            Printable result with chi-squared and Iman-Davenport
            statistics, average ranks, and significance verdict.
        """
        from surfaces.benchmark._statistics import compute_friedman

        return compute_friedman(self._bench._traces, alpha, at_cu)

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
                "catch": self._bench._catch,
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
    """Benchmark visualization via Plotly.

    Accessed via ``bench.plot``. All methods return a
    ``plotly.graph_objects.Figure`` that can be displayed with
    ``fig.show()`` or rendered automatically in Jupyter notebooks.

    Requires the ``viz`` extra (``pip install surfaces[viz]``).
    """

    def __init__(self, benchmark: Benchmark) -> None:
        self._bench = benchmark
        self._target_cache: dict[str, float] | None = None
        self._best_known_warned: bool = False

    def _check_plotly(self) -> None:
        try:
            import plotly  # noqa: F401
        except ImportError:
            raise ImportError(
                "Plotly is required for benchmark plots. Install it with: pip install surfaces[viz]"
            )

    def _resolve_targets(self, precision: float) -> dict[str, float]:
        """Compute target score per function: f_global + precision, or best-known fallback.

        Results are cached per Benchmark instance. The best-known fallback
        uses the best score observed across all optimizers and seeds for
        that function. A warning is emitted once listing which functions
        use the fallback.
        """
        if self._target_cache is not None:
            cached_precision = getattr(self, "_cached_precision", None)
            if cached_precision == precision:
                return self._target_cache

        targets = {}
        best_known_funcs = []

        for func_cls in self._bench._functions:
            name = func_cls.__name__
            if func_cls.f_global is not None:
                targets[name] = func_cls.f_global + precision
            else:
                scores = [
                    t.best_score
                    for (f, _, _), t in self._bench._traces.items()
                    if f == name and t.best_score is not None
                ]
                if not scores:
                    continue
                best = min(scores)
                targets[name] = best + precision
                best_known_funcs.append(f"{name} (best observed: {best:.6f})")

        if best_known_funcs and not self._best_known_warned:
            warnings.warn(
                "f_global is unknown for: "
                + ", ".join(best_known_funcs)
                + ". Using best observed score as reference. "
                "ECDF results for these functions are relative to "
                "this benchmark run, not absolute.",
                stacklevel=3,
            )
            self._best_known_warned = True

        self._target_cache = targets
        self._cached_precision = precision
        return targets

    def ecdf(
        self,
        precision: float | list[float] = 1.0,
        log_x: bool = True,
    ) -> Any:
        """Empirical Cumulative Distribution Function of running times.

        Shows for each optimizer what fraction of (function, seed) problems
        it solved within a given CU budget. A problem counts as "solved"
        when ``best_so_far <= target``, where target is
        ``f_global + precision`` (or best-known + precision as fallback).

        Parameters
        ----------
        precision : float or list[float]
            Target precision(s). A list produces stacked subplots, one
            per precision level, useful for comparing difficulty grades
            like ``[1.0, 0.1, 0.01]``.
        log_x : bool
            Logarithmic x-axis. Standard in benchmark literature.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        self._check_plotly()
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if not self._bench._traces:
            raise ValueError("No traces recorded yet. Call run() first.")

        precisions = precision if isinstance(precision, list) else [precision]

        if len(precisions) == 1:
            fig = go.Figure()
            self._add_ecdf_traces(fig, precisions[0])
            fig.update_layout(
                title=f"ECDF (precision={precisions[0]})",
                xaxis_title="CU Budget",
                yaxis_title="Fraction of problems solved",
                yaxis_range=[0, 1.05],
                xaxis_type="log" if log_x else "linear",
                legend_title="Optimizer",
                template="plotly_white",
            )
        else:
            fig = make_subplots(
                rows=len(precisions),
                cols=1,
                shared_xaxes=True,
                subplot_titles=[f"precision={p}" for p in precisions],
                vertical_spacing=0.08,
            )
            for i, p in enumerate(precisions, start=1):
                self._add_ecdf_traces(fig, p, row=i, col=1, show_legend=(i == 1))
                fig.update_yaxes(
                    title_text="Fraction solved" if i == len(precisions) else "",
                    range=[0, 1.05],
                    row=i,
                    col=1,
                )
                if log_x:
                    fig.update_xaxes(type="log", row=i, col=1)

            fig.update_xaxes(title_text="CU Budget", row=len(precisions), col=1)
            fig.update_layout(
                title="ECDF across precision levels",
                legend_title="Optimizer",
                template="plotly_white",
                height=300 * len(precisions),
            )

        return fig

    def _add_ecdf_traces(
        self,
        fig: Any,
        precision: float,
        row: int | None = None,
        col: int | None = None,
        show_legend: bool = True,
    ) -> None:
        """Add ECDF step-function traces for one precision level."""
        import plotly.graph_objects as go

        targets = self._resolve_targets(precision)
        opt_names = sorted({k[1] for k in self._bench._traces})

        for opt_name in opt_names:
            running_times = []
            for func_name in targets:
                traces_for_pair = [
                    t
                    for (f, o, _), t in self._bench._traces.items()
                    if f == func_name and o == opt_name
                ]
                for trace in traces_for_pair:
                    rt = self._running_time(trace, targets[func_name])
                    running_times.append(rt)

            if not running_times:
                continue

            n_problems = len(running_times)
            solved = sorted(rt for rt in running_times if np.isfinite(rt))

            x_vals = [0.0]
            y_vals = [0.0]
            for i, rt in enumerate(solved):
                x_vals.extend([rt, rt])
                y_vals.extend([i / n_problems, (i + 1) / n_problems])

            if solved:
                max_cu = max(t.total_cu for t in self._bench._traces.values() if t.total_cu > 0)
                x_vals.append(max_cu)
                y_vals.append(len(solved) / n_problems)

            kwargs = dict(row=row, col=col) if row is not None else {}
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name=opt_name,
                    showlegend=show_legend,
                ),
                **kwargs,
            )

    @staticmethod
    def _running_time(trace: Trace, target: float) -> float:
        """CU at which the trace first reaches the target, or inf."""
        for record in trace:
            if record.best_so_far <= target:
                return record.cumulative_cu
        return float("inf")

    def convergence(
        self,
        function: str,
        band: str = "iqr",
        center: str = "median",
        log_y: bool = False,
    ) -> Any:
        """Convergence plot for a single function across all optimizers.

        Shows how quickly each optimizer converges by plotting the center
        line (median or mean of best_so_far across seeds) with an
        optional uncertainty band.

        Parameters
        ----------
        function : str
            Function name to plot.
        band : str or None
            Uncertainty band style: ``"iqr"`` (25th-75th percentile),
            ``"minmax"``, ``"std"`` (center +/- 1 standard deviation),
            or ``None`` to hide the band.
        center : str
            Center line statistic: ``"median"`` or ``"mean"``.
        log_y : bool
            Logarithmic y-axis.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        self._check_plotly()
        import plotly.graph_objects as go

        if not self._bench._traces:
            raise ValueError("No traces recorded yet. Call run() first.")

        func_traces = {
            (f, o, s): t for (f, o, s), t in self._bench._traces.items() if f == function
        }
        if not func_traces:
            available = sorted({k[0] for k in self._bench._traces})
            raise ValueError(
                f"No traces for function '{function}'. Available: {', '.join(available)}"
            )

        opt_names = sorted({o for (_, o, _) in func_traces})
        fig = go.Figure()

        for opt_name in opt_names:
            seed_traces = [t for (_, o, _), t in func_traces.items() if o == opt_name]
            cu_grid, center_vals, lower, upper = self._interpolate_convergence(
                seed_traces, center, band
            )
            if cu_grid is None:
                continue

            if band is not None and lower is not None:
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([cu_grid, cu_grid[::-1]]),
                        y=np.concatenate([upper, lower[::-1]]),
                        fill="toself",
                        fillcolor=None,
                        line=dict(width=0),
                        opacity=0.2,
                        name=f"{opt_name} ({band})",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=cu_grid,
                    y=center_vals,
                    mode="lines",
                    name=opt_name,
                )
            )

        fig.update_layout(
            title=f"Convergence: {function}",
            xaxis_title="CU Budget",
            yaxis_title="Best score",
            yaxis_type="log" if log_y else "linear",
            legend_title="Optimizer",
            template="plotly_white",
        )

        return fig

    @staticmethod
    def _interpolate_convergence(
        seed_traces: list[Trace],
        center: str,
        band: str | None,
    ) -> tuple[Any, Any, Any, Any]:
        """Interpolate best_so_far across seeds onto a common CU grid.

        Returns (cu_grid, center_values, lower_band, upper_band).
        lower_band and upper_band are None when band is None.
        """
        if not seed_traces:
            return None, None, None, None

        all_cu = np.unique(
            np.concatenate([np.array([r.cumulative_cu for r in t]) for t in seed_traces])
        )

        if len(all_cu) == 0:
            return None, None, None, None

        grid_size = min(500, len(all_cu))
        cu_grid = np.linspace(all_cu[0], all_cu[-1], grid_size)

        interpolated = np.full((len(seed_traces), grid_size), np.nan)
        for i, trace in enumerate(seed_traces):
            cu_points = np.array([r.cumulative_cu for r in trace])
            best_points = np.array([r.best_so_far for r in trace])
            indices = np.searchsorted(cu_points, cu_grid, side="right") - 1
            valid = indices >= 0
            interpolated[i, valid] = best_points[indices[valid]]

        valid_mask = ~np.all(np.isnan(interpolated), axis=0)
        cu_grid = cu_grid[valid_mask]
        interpolated = interpolated[:, valid_mask]

        if cu_grid.size == 0:
            return None, None, None, None

        if center == "median":
            center_vals = np.nanmedian(interpolated, axis=0)
        else:
            center_vals = np.nanmean(interpolated, axis=0)

        lower, upper = None, None
        if band == "iqr":
            lower = np.nanpercentile(interpolated, 25, axis=0)
            upper = np.nanpercentile(interpolated, 75, axis=0)
        elif band == "minmax":
            lower = np.nanmin(interpolated, axis=0)
            upper = np.nanmax(interpolated, axis=0)
        elif band == "std":
            std = np.nanstd(interpolated, axis=0)
            lower = center_vals - std
            upper = center_vals + std

        return cu_grid, center_vals, lower, upper

    def cd_diagram(
        self,
        at_cu: float | None = None,
        alpha: float = 0.05,
        correction: str | None = "holm",
        title: str | None = None,
        width: float = 8.0,
    ) -> Any:
        """Critical Difference diagram comparing optimizer ranks.

        Visualizes average ranks on a horizontal axis with thick bars
        connecting groups of optimizers that are not statistically
        distinguishable (Demsar, 2006).

        Average ranks are computed with proper tied-rank handling
        using only functions where all optimizers produced results
        (complete blocks), matching the Friedman test methodology.

        Requires matplotlib (``pip install surfaces[viz]``).

        Parameters
        ----------
        at_cu : float, optional
            Evaluate scores at this CU budget.
        alpha : float, default=0.05
            Significance level for clique detection.
        correction : str or None, default="holm"
            P-value correction for pairwise Wilcoxon tests.
        title : str, optional
            Figure title. Defaults to include the alpha value.
        width : float, default=8.0
            Figure width in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from surfaces.benchmark._cd_diagram import render_cd_diagram
        from surfaces.benchmark._statistics import (
            _build_score_matrix,
            _compute_avg_ranks,
            compute_ranking,
        )

        traces = self._bench._traces
        if not traces:
            raise ValueError("No traces recorded yet. Call run() first.")

        ranking = compute_ranking(traces, alpha, at_cu, correction)

        if len(ranking.entries) < 2:
            raise ValueError("CD diagram requires at least 2 optimizers with results")

        func_names, opt_names, scores = _build_score_matrix(traces, at_cu)
        complete_funcs = [f for f in func_names if all(o in scores[f] for o in opt_names)]
        if not complete_funcs:
            complete_funcs = func_names

        avg_ranks = _compute_avg_ranks(complete_funcs, opt_names, scores)

        return render_cd_diagram(
            avg_ranks=avg_ranks,
            pvalues=ranking.pvalues,
            alpha=alpha,
            title=title,
            width=width,
        )

    def __repr__(self) -> str:
        return "PlotAccessor(.ecdf(), .convergence(), .cd_diagram())"
