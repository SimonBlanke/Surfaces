"""Statistical analysis for benchmark results.

Provides ERT (Expected Running Time) and ranking computations
with structured result objects that are printable, subscriptable,
and exportable to DataFrames.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

from surfaces.benchmark._trace import Trace


@dataclass(frozen=True)
class ERTEntry:
    """ERT result for a single (function, optimizer) pair."""

    ert_cu: float
    solved: int
    total: int
    median_cu: float
    individual_cu: tuple[float, ...]

    @property
    def success_rate(self) -> float:
        return self.solved / self.total if self.total > 0 else 0.0


class ERTTable:
    """Expected Running Time results across functions and optimizers.

    Subscriptable by function name, then optimizer name::

        ert = bench.results.ert()
        entry = ert["AckleyFunction"]["HillClimbing"]
        print(entry.ert_cu, entry.solved, entry.total)
    """

    def __init__(
        self,
        data: dict[str, dict[str, ERTEntry]],
        precision: float,
        function_names: list[str],
        optimizer_names: list[str],
    ):
        self._data = data
        self.precision = precision
        self._function_names = function_names
        self._optimizer_names = optimizer_names

    def __getitem__(self, function: str) -> dict[str, ERTEntry]:
        return self._data[function]

    def __contains__(self, function: str) -> bool:
        return function in self._data

    def to_dataframe(self) -> Any:
        import pandas as pd

        rows = []
        for func in self._function_names:
            for opt in self._optimizer_names:
                if func in self._data and opt in self._data[func]:
                    e = self._data[func][opt]
                    rows.append(
                        {
                            "function": func,
                            "optimizer": opt,
                            "ert_cu": e.ert_cu,
                            "median_cu": e.median_cu,
                            "solved": e.solved,
                            "total": e.total,
                            "success_rate": e.success_rate,
                        }
                    )
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        opt_names = self._optimizer_names
        func_names = self._function_names

        opt_col_w = max(14, *(len(n) + 2 for n in opt_names))
        func_col_w = max(18, *(len(n) for n in func_names))

        header = f"{'Function':<{func_col_w}}"
        for opt in opt_names:
            header += f"  {opt:>{opt_col_w}}"
        lines = [
            f"ERT (target = f_global + {self.precision})",
            "",
            header,
            "\u2500" * len(header),
        ]

        for func in func_names:
            row = f"{func:<{func_col_w}}"
            for opt in opt_names:
                entry = self._data.get(func, {}).get(opt)
                if entry is None:
                    cell = "n/a"
                elif math.isinf(entry.ert_cu):
                    cell = f"inf ({entry.solved}/{entry.total})"
                else:
                    cell = f"{entry.ert_cu:,.0f} ({entry.solved}/{entry.total})"
                row += f"  {cell:>{opt_col_w}}"
            lines.append(row)

        lines.append("")
        lines.append("Values in CU. (solved/total) = seeds reaching target.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n = sum(1 for f in self._data.values() for _ in f.values())
        return f"ERTTable({n} entries, precision={self.precision})"


@dataclass(frozen=True)
class RankingEntry:
    """Ranking result for a single optimizer."""

    optimizer: str
    rank: float
    mean_normalized: float


class RankingTable:
    """Optimizer ranking with pairwise statistical tests.

    Subscriptable by optimizer name::

        ranking = bench.results.ranking()
        entry = ranking["HillClimbing"]
        print(entry.rank, entry.mean_normalized)
        print(ranking.pvalues)
    """

    def __init__(
        self,
        entries: list[RankingEntry],
        pvalues: dict[tuple[str, str], float],
        alpha: float,
        normalized_scores: dict[str, dict[str, float]],
    ):
        self.entries = entries
        self.pvalues = pvalues
        self.alpha = alpha
        self._normalized_scores = normalized_scores
        self._by_name = {e.optimizer: e for e in entries}

    def __getitem__(self, optimizer: str) -> RankingEntry:
        return self._by_name[optimizer]

    def __contains__(self, optimizer: str) -> bool:
        return optimizer in self._by_name

    def to_dataframe(self) -> Any:
        import pandas as pd

        rows = [
            {
                "optimizer": e.optimizer,
                "rank": e.rank,
                "mean_normalized": e.mean_normalized,
            }
            for e in self.entries
        ]
        return pd.DataFrame(rows)

    def pvalues_dataframe(self) -> Any:
        """Pairwise Wilcoxon p-values as a square DataFrame."""
        import pandas as pd

        names = [e.optimizer for e in self.entries]
        matrix = {}
        for a in names:
            row = {}
            for b in names:
                if a == b:
                    row[b] = float("nan")
                else:
                    key = (a, b) if (a, b) in self.pvalues else (b, a)
                    row[b] = self.pvalues.get(key, float("nan"))
            matrix[a] = row
        return pd.DataFrame(matrix, index=names)

    def __str__(self) -> str:
        opt_names = [e.optimizer for e in self.entries]
        name_w = max(20, *(len(n) + 2 for n in opt_names))

        lines = [
            "Optimizer Ranking",
            "",
            f"{'Rank':>4}  {'Optimizer':<{name_w}}  {'Mean norm.':>10}  {'Mean rank':>10}",
            "\u2500" * (4 + 2 + name_w + 2 + 10 + 2 + 10),
        ]
        for e in self.entries:
            lines.append(
                f"{e.rank:>4.1f}  {e.optimizer:<{name_w}}  {e.mean_normalized:>10.4f}  {e.rank:>10.1f}"
            )

        if self.pvalues:
            lines.append("")
            lines.append(f"Pairwise Wilcoxon signed-rank (alpha={self.alpha})")
            lines.append("")

            col_w = max(12, *(len(n) + 2 for n in opt_names))
            header = " " * name_w
            for n in opt_names:
                header += f"  {n:>{col_w}}"
            lines.append(header)
            lines.append("\u2500" * len(header))

            for a in opt_names:
                row = f"{a:<{name_w}}"
                for b in opt_names:
                    if a == b:
                        cell = "\u2014"
                    else:
                        key = (a, b) if (a, b) in self.pvalues else (b, a)
                        p = self.pvalues.get(key)
                        if p is None:
                            cell = "n/a"
                        elif p < self.alpha:
                            cell = f"{p:.3f}*"
                        else:
                            cell = f"{p:.3f}"
                    row += f"  {cell:>{col_w}}"
                lines.append(row)

            lines.append("")
            lines.append(f"* = significant at alpha={self.alpha}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"RankingTable({len(self.entries)} optimizers, alpha={self.alpha})"


def compute_ert(
    traces: dict[tuple[str, str, int], Trace],
    optimal_scores: dict[str, float | None],
    precision: float,
    targets: dict[str, float] | None = None,
) -> ERTTable:
    """Compute Expected Running Time for all (function, optimizer) pairs.

    For each trace, finds the first cumulative_cu where
    best_so_far <= target. ERT is computed as the COCO standard:
    sum(all running times including inf) / number_of_successful_runs.
    """
    func_names = sorted({k[0] for k in traces})
    opt_names = sorted({k[1] for k in traces})

    resolved_targets: dict[str, float] = {}
    for func in func_names:
        if targets and func in targets:
            resolved_targets[func] = targets[func]
        elif optimal_scores.get(func) is not None:
            resolved_targets[func] = optimal_scores[func] + precision
        else:
            resolved_targets[func] = precision

    data: dict[str, dict[str, ERTEntry]] = {}
    for func in func_names:
        data[func] = {}
        target = resolved_targets[func]

        for opt in opt_names:
            matching = [t for (f, o, _s), t in traces.items() if f == func and o == opt]
            if not matching:
                continue

            running_times: list[float] = []
            for trace in matching:
                rt = _running_time(trace, target)
                running_times.append(rt)

            solved_times = [rt for rt in running_times if not math.isinf(rt)]
            n_solved = len(solved_times)
            n_total = len(running_times)

            if n_solved == 0:
                ert_cu = float("inf")
                median_cu = float("inf")
            else:
                total_cu = sum(
                    rt if not math.isinf(rt) else matching[i].total_cu
                    for i, rt in enumerate(running_times)
                )
                ert_cu = total_cu / n_solved
                sorted_times = sorted(solved_times)
                mid = len(sorted_times) // 2
                if len(sorted_times) % 2 == 0:
                    median_cu = (sorted_times[mid - 1] + sorted_times[mid]) / 2
                else:
                    median_cu = sorted_times[mid]

            data[func][opt] = ERTEntry(
                ert_cu=ert_cu,
                solved=n_solved,
                total=n_total,
                median_cu=median_cu,
                individual_cu=tuple(running_times),
            )

    return ERTTable(data, precision, func_names, opt_names)


def compute_ranking(
    traces: dict[tuple[str, str, int], Trace],
    alpha: float = 0.05,
    at_cu: float | None = None,
) -> RankingTable:
    """Rank optimizers by normalized scores with Wilcoxon tests.

    Normalization is per-function: 0 = worst observed, 1 = best observed.
    The ranking uses mean normalized scores averaged over seeds,
    then ranks across functions.
    """
    func_names = sorted({k[0] for k in traces})
    opt_names = sorted({k[1] for k in traces})

    mean_scores: dict[str, dict[str, float]] = {}
    for func in func_names:
        func_scores: dict[str, list[float]] = {}
        for opt in opt_names:
            matching = [t for (f, o, _s), t in traces.items() if f == func and o == opt]
            if not matching:
                continue
            scores = []
            for t in matching:
                if at_cu is not None:
                    s = t.score_at_cu(at_cu)
                else:
                    s = t.best_score
                if s is not None:
                    scores.append(s)
            if scores:
                func_scores[opt] = scores

        if len(func_scores) < 2:
            continue

        all_scores = [s for seeds in func_scores.values() for s in seeds]
        worst = max(all_scores)
        best = min(all_scores)
        score_range = worst - best

        if func not in mean_scores:
            mean_scores[func] = {}

        for opt, seeds in func_scores.items():
            mean_raw = sum(seeds) / len(seeds)
            if score_range > 0:
                normalized = 1.0 - (mean_raw - best) / score_range
            else:
                normalized = 1.0
            mean_scores[func][opt] = normalized

    ranks_per_func: dict[str, dict[str, float]] = {}
    for func, opt_scores in mean_scores.items():
        sorted_opts = sorted(opt_scores.items(), key=lambda x: -x[1])
        ranks_per_func[func] = {}
        for rank_idx, (opt, _score) in enumerate(sorted_opts, 1):
            ranks_per_func[func][opt] = float(rank_idx)

    agg: dict[str, list[float]] = {opt: [] for opt in opt_names}
    norm_agg: dict[str, list[float]] = {opt: [] for opt in opt_names}
    for func in mean_scores:
        for opt in opt_names:
            if opt in ranks_per_func.get(func, {}):
                agg[opt].append(ranks_per_func[func][opt])
            if opt in mean_scores.get(func, {}):
                norm_agg[opt].append(mean_scores[func][opt])

    entries = []
    for opt in opt_names:
        if agg[opt]:
            mean_rank = sum(agg[opt]) / len(agg[opt])
            mean_norm = sum(norm_agg[opt]) / len(norm_agg[opt])
            entries.append(RankingEntry(opt, mean_rank, mean_norm))
    entries.sort(key=lambda e: e.rank)

    pvalues: dict[tuple[str, str], float] = {}
    n_functions = len(mean_scores)
    if n_functions < 6:
        warnings.warn(
            f"Only {n_functions} functions. Wilcoxon test needs at least 6 "
            f"for meaningful p-values. Results may be unreliable.",
            stacklevel=3,
        )

    if n_functions >= 2:
        from scipy.stats import wilcoxon

        for i, a in enumerate(opt_names):
            for b in opt_names[i + 1 :]:
                scores_a = []
                scores_b = []
                for func in mean_scores:
                    if a in mean_scores[func] and b in mean_scores[func]:
                        scores_a.append(mean_scores[func][a])
                        scores_b.append(mean_scores[func][b])
                if len(scores_a) >= 2:
                    try:
                        _, p = wilcoxon(scores_a, scores_b)
                        pvalues[(a, b)] = float(p)
                    except ValueError:
                        pvalues[(a, b)] = float("nan")

    return RankingTable(entries, pvalues, alpha, mean_scores)


def _running_time(trace: Trace, target: float) -> float:
    """CU at which the trace first reaches the target score."""
    for record in trace:
        if record.best_so_far <= target:
            return record.cumulative_cu
    return float("inf")
