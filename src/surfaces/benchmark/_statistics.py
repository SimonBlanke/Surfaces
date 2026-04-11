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


def _build_score_matrix(
    traces: dict[tuple[str, str, int], Trace],
    at_cu: float | None = None,
) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
    """Build per-function normalized scores from traces.

    Returns (func_names, opt_names, scores) where
    scores[func][opt] is in [0, 1] with 1 = best observed.
    Only functions with at least 2 optimizers are included.
    """
    all_func_names = sorted({k[0] for k in traces})
    all_opt_names = sorted({k[1] for k in traces})

    scores: dict[str, dict[str, float]] = {}

    for func in all_func_names:
        func_scores: dict[str, list[float]] = {}
        for opt in all_opt_names:
            matching = [t for (f, o, _s), t in traces.items() if f == func and o == opt]
            if not matching:
                continue
            vals = []
            for t in matching:
                s = t.score_at_cu(at_cu) if at_cu is not None else t.best_score
                if s is not None:
                    vals.append(s)
            if vals:
                func_scores[opt] = vals

        if len(func_scores) < 2:
            continue

        all_vals = [s for seeds in func_scores.values() for s in seeds]
        worst, best = max(all_vals), min(all_vals)
        score_range = worst - best

        scores[func] = {}
        for opt, seeds in func_scores.items():
            mean_raw = sum(seeds) / len(seeds)
            scores[func][opt] = (1.0 - (mean_raw - best) / score_range) if score_range > 0 else 1.0

    return sorted(scores.keys()), all_opt_names, scores


def _compute_avg_ranks(
    func_names: list[str],
    opt_names: list[str],
    scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Average rank per optimizer across functions with tied-rank handling.

    Within each function, optimizers are ranked by normalized score
    (higher = better = lower rank number). Tied scores receive the
    mean of the ranks they span.
    """
    rank_sums: dict[str, float] = {opt: 0.0 for opt in opt_names}
    count: dict[str, int] = {opt: 0 for opt in opt_names}

    for func in func_names:
        present = [(opt, scores[func][opt]) for opt in opt_names if opt in scores[func]]
        present.sort(key=lambda x: -x[1])

        i = 0
        while i < len(present):
            j = i + 1
            while j < len(present) and present[j][1] == present[i][1]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for idx in range(i, j):
                rank_sums[present[idx][0]] += avg_rank
                count[present[idx][0]] += 1
            i = j

    return {opt: rank_sums[opt] / count[opt] for opt in opt_names if count[opt] > 0}


def _holm_correction(
    pvalues: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """Holm step-down correction for multiple pairwise comparisons.

    Adjusts p-values upward so they can be compared directly against
    alpha while controlling the family-wise error rate. Strictly more
    powerful than Bonferroni.
    """
    valid = [(k, v) for k, v in pvalues.items() if not math.isnan(v)]
    valid.sort(key=lambda x: x[1])

    m = len(valid)
    corrected: dict[tuple[str, str], float] = {}

    prev = 0.0
    for i, (key, p) in enumerate(valid):
        adjusted = min(1.0, p * (m - i))
        adjusted = max(adjusted, prev)
        corrected[key] = adjusted
        prev = adjusted

    for k, v in pvalues.items():
        if math.isnan(v):
            corrected[k] = v

    return corrected


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
        correction: str | None = None,
    ):
        self.entries = entries
        self.pvalues = pvalues
        self.alpha = alpha
        self._normalized_scores = normalized_scores
        self._by_name = {e.optimizer: e for e in entries}
        self.correction = correction

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
            test_label = "Pairwise Wilcoxon signed-rank"
            if self.correction:
                test_label += f" with {self.correction.title()} correction"
            lines.append(f"{test_label} (alpha={self.alpha})")
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
        corr = f", correction={self.correction!r}" if self.correction else ""
        return f"RankingTable({len(self.entries)} optimizers, alpha={self.alpha}{corr})"


@dataclass(frozen=True)
class FriedmanResult:
    """Result of the Friedman omnibus test for comparing multiple optimizers.

    The Friedman test checks whether at least one optimizer differs
    significantly. If ``significant`` is True, proceed with post-hoc
    pairwise tests. If False, observed differences are not statistically
    supported.

    The Iman-Davenport variant uses an F-distribution and is less
    conservative than the chi-squared approximation.
    """

    chi2_statistic: float
    chi2_p_value: float
    f_statistic: float
    f_p_value: float
    n_functions: int
    n_optimizers: int
    alpha: float
    avg_ranks: dict[str, float]

    @property
    def significant(self) -> bool:
        """Whether the Iman-Davenport test rejects the null hypothesis."""
        return self.f_p_value < self.alpha

    def __str__(self) -> str:
        lines = [
            f"Friedman Test (alpha={self.alpha})",
            "",
            f"  N functions:      {self.n_functions}",
            f"  k optimizers:     {self.n_optimizers}",
            "",
            f"  Chi-squared:      {self.chi2_statistic:.4f}  (p={self.chi2_p_value:.6f})",
            f"  Iman-Davenport F: {self.f_statistic:.4f}  (p={self.f_p_value:.6f})",
            "",
        ]
        if self.significant:
            lines.append(f"  Result: Significant (p={self.f_p_value:.6f} < {self.alpha})")
            lines.append("  At least one optimizer differs. Proceed with post-hoc tests.")
        else:
            lines.append(f"  Result: Not significant (p={self.f_p_value:.6f} >= {self.alpha})")
            lines.append("  No evidence that optimizers differ.")

        lines.append("")
        lines.append("  Average ranks:")
        for opt, rank in sorted(self.avg_ranks.items(), key=lambda x: x[1]):
            lines.append(f"    {rank:5.2f}  {opt}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return (
            f"FriedmanResult(p={self.f_p_value:.6f}, {sig}, "
            f"k={self.n_optimizers}, N={self.n_functions})"
        )


def compute_friedman(
    traces: dict[tuple[str, str, int], Trace],
    alpha: float = 0.05,
    at_cu: float | None = None,
) -> FriedmanResult:
    """Friedman omnibus test on benchmark traces.

    Requires at least 3 optimizers and 3 functions where all
    optimizers produced results. Uses normalized scores.

    The Iman-Davenport correction replaces the chi-squared
    approximation with an F-distribution, giving a less conservative
    (more powerful) test.
    """
    func_names, opt_names, scores = _build_score_matrix(traces, at_cu)

    complete_funcs = [f for f in func_names if all(o in scores[f] for o in opt_names)]

    k = len(opt_names)
    n = len(complete_funcs)

    if k < 3:
        raise ValueError(f"Friedman test requires at least 3 optimizers, got {k}")
    if n < 3:
        raise ValueError(
            f"Friedman test requires at least 3 functions with complete data, "
            f"got {n}. Ensure all optimizers produce results for each function."
        )

    from scipy.stats import friedmanchisquare

    arrays = [[scores[f][opt] for f in complete_funcs] for opt in opt_names]
    chi2, p_chi2 = friedmanchisquare(*arrays)

    denom = n * (k - 1) - chi2
    if denom <= 0:
        f_stat = float("inf")
        p_f = 0.0
    else:
        f_stat = ((n - 1) * chi2) / denom
        from scipy.stats import f as f_dist

        p_f = 1.0 - f_dist.cdf(f_stat, k - 1, (k - 1) * (n - 1))

    avg_ranks = _compute_avg_ranks(complete_funcs, opt_names, scores)

    return FriedmanResult(
        chi2_statistic=float(chi2),
        chi2_p_value=float(p_chi2),
        f_statistic=float(f_stat),
        f_p_value=float(p_f),
        n_functions=n,
        n_optimizers=k,
        alpha=alpha,
        avg_ranks=avg_ranks,
    )


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
    correction: str | None = "holm",
) -> RankingTable:
    """Rank optimizers by normalized scores with pairwise statistical tests.

    Normalization is per-function: 0 = worst observed, 1 = best observed.
    Ranks use tied-rank averaging within each function, then are
    averaged across functions. Pairwise Wilcoxon signed-rank tests
    assess significance, with optional Holm correction for multiple
    comparisons.

    Parameters
    ----------
    correction : str or None
        ``"holm"`` applies Holm step-down correction (recommended).
        ``None`` returns raw uncorrected p-values.
    """
    func_names, opt_names, mean_scores = _build_score_matrix(traces, at_cu)

    avg_ranks = _compute_avg_ranks(func_names, opt_names, mean_scores)

    norm_agg: dict[str, list[float]] = {opt: [] for opt in opt_names}
    for func in func_names:
        for opt in opt_names:
            if opt in mean_scores.get(func, {}):
                norm_agg[opt].append(mean_scores[func][opt])

    entries = []
    for opt in opt_names:
        if opt in avg_ranks:
            mean_norm = sum(norm_agg[opt]) / len(norm_agg[opt]) if norm_agg[opt] else 0.0
            entries.append(RankingEntry(opt, avg_ranks[opt], mean_norm))
    entries.sort(key=lambda e: e.rank)

    pvalues: dict[tuple[str, str], float] = {}
    n_functions = len(func_names)
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
                for func in func_names:
                    if a in mean_scores[func] and b in mean_scores[func]:
                        scores_a.append(mean_scores[func][a])
                        scores_b.append(mean_scores[func][b])
                if len(scores_a) >= 2:
                    try:
                        _, p = wilcoxon(scores_a, scores_b)
                        pvalues[(a, b)] = float(p)
                    except ValueError:
                        pvalues[(a, b)] = float("nan")

    if correction is not None and correction != "holm":
        raise ValueError(f"Unknown correction: {correction!r}. Use 'holm' or None.")
    if correction == "holm" and pvalues:
        pvalues = _holm_correction(pvalues)

    return RankingTable(entries, pvalues, alpha, mean_scores, correction)


def _running_time(trace: Trace, target: float) -> float:
    """CU at which the trace first reaches the target score."""
    for record in trace:
        if record.best_so_far <= target:
            return record.cumulative_cu
    return float("inf")
