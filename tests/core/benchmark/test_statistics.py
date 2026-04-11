"""Tests for benchmark statistical analysis: ERT and ranking."""

from __future__ import annotations

import math

import pytest

from surfaces.benchmark._statistics import (
    ERTEntry,
    ERTTable,
    RankingTable,
    _running_time,
    compute_ert,
    compute_ranking,
)
from surfaces.benchmark._trace import EvalRecord, Trace


def _trace(scores: list[float], cu_per_step: float = 10.0) -> Trace:
    """Build a Trace from a list of scores with uniform CU steps."""
    trace = Trace()
    cumulative = 0.0
    best = float("inf")
    for score in scores:
        cumulative += cu_per_step
        best = min(best, score)
        trace.append(
            EvalRecord(
                params={"x0": 0.0},
                score=score,
                eval_cu=cu_per_step * 0.9,
                overhead_cu=cu_per_step * 0.1,
                cumulative_cu=cumulative,
                best_so_far=best,
                wall_seconds=0.001,
            )
        )
    return trace


class TestRunningTime:
    def test_target_reached_first_step(self):
        trace = _trace([0.5, 0.3, 0.1])
        assert _running_time(trace, target=1.0) == 10.0

    def test_target_reached_later(self):
        trace = _trace([5.0, 3.0, 0.8])
        assert _running_time(trace, target=1.0) == 30.0

    def test_target_never_reached(self):
        trace = _trace([5.0, 3.0, 2.0])
        assert math.isinf(_running_time(trace, target=1.0))

    def test_empty_trace(self):
        trace = Trace()
        assert math.isinf(_running_time(trace, target=1.0))

    def test_exact_target(self):
        trace = _trace([2.0, 1.0, 0.5])
        assert _running_time(trace, target=1.0) == 20.0


class TestComputeERT:
    def setup_method(self):
        self.traces = {
            ("Ackley", "OptA", 0): _trace([5.0, 2.0, 0.5]),
            ("Ackley", "OptA", 1): _trace([5.0, 3.0, 1.5]),
            ("Ackley", "OptA", 2): _trace([5.0, 4.0, 3.0]),
            ("Ackley", "OptB", 0): _trace([5.0, 0.1, 0.05]),
            ("Ackley", "OptB", 1): _trace([5.0, 0.2, 0.1]),
            ("Ackley", "OptB", 2): _trace([5.0, 0.3, 0.15]),
        }
        self.optimal = {"Ackley": 0.0}

    def test_basic_ert(self):
        ert = compute_ert(self.traces, self.optimal, precision=1.0)
        entry_a = ert["Ackley"]["OptA"]
        assert entry_a.solved == 1
        assert entry_a.total == 3

        entry_b = ert["Ackley"]["OptB"]
        assert entry_b.solved == 3
        assert entry_b.total == 3
        assert entry_b.ert_cu < entry_a.ert_cu

    def test_all_solved(self):
        ert = compute_ert(self.traces, self.optimal, precision=5.0)
        entry = ert["Ackley"]["OptA"]
        assert entry.solved == 3
        assert entry.total == 3
        assert not math.isinf(entry.ert_cu)

    def test_none_solved(self):
        ert = compute_ert(self.traces, self.optimal, precision=0.001)
        entry = ert["Ackley"]["OptA"]
        assert entry.solved == 0
        assert math.isinf(entry.ert_cu)
        assert math.isinf(entry.median_cu)

    def test_custom_targets(self):
        ert = compute_ert(self.traces, self.optimal, precision=1.0, targets={"Ackley": 4.5})
        entry = ert["Ackley"]["OptA"]
        assert entry.solved == 3

    def test_missing_optimal_uses_precision_only(self):
        ert = compute_ert(self.traces, {}, precision=1.0)
        entry = ert["Ackley"]["OptA"]
        assert entry.solved == 1

    def test_subscript_access(self):
        ert = compute_ert(self.traces, self.optimal, precision=1.0)
        assert "Ackley" in ert
        assert isinstance(ert["Ackley"]["OptA"], ERTEntry)

    def test_str_output(self):
        ert = compute_ert(self.traces, self.optimal, precision=1.0)
        text = str(ert)
        assert "ERT" in text
        assert "Ackley" in text
        assert "OptA" in text

    def test_to_dataframe(self):
        ert = compute_ert(self.traces, self.optimal, precision=1.0)
        df = ert.to_dataframe()
        assert "ert_cu" in df.columns
        assert "success_rate" in df.columns
        assert len(df) == 2

    def test_individual_cu_stored(self):
        ert = compute_ert(self.traces, self.optimal, precision=1.0)
        entry = ert["Ackley"]["OptA"]
        assert len(entry.individual_cu) == 3

    def test_median_computation(self):
        ert = compute_ert(self.traces, self.optimal, precision=1.0)
        entry_b = ert["Ackley"]["OptB"]
        assert entry_b.solved == 3
        assert entry_b.median_cu == 20.0

    def test_ert_entry_success_rate(self):
        entry = ERTEntry(ert_cu=100, solved=3, total=5, median_cu=90, individual_cu=(80, 90, 100))
        assert entry.success_rate == pytest.approx(0.6)


class TestComputeRanking:
    def setup_method(self):
        self.traces = {
            ("F1", "OptA", 0): _trace([5.0, 1.0]),
            ("F1", "OptA", 1): _trace([5.0, 1.2]),
            ("F1", "OptB", 0): _trace([5.0, 3.0]),
            ("F1", "OptB", 1): _trace([5.0, 3.5]),
            ("F2", "OptA", 0): _trace([10.0, 8.0]),
            ("F2", "OptA", 1): _trace([10.0, 7.0]),
            ("F2", "OptB", 0): _trace([10.0, 2.0]),
            ("F2", "OptB", 1): _trace([10.0, 3.0]),
        }

    def test_basic_ranking(self):
        ranking = compute_ranking(self.traces)
        assert len(ranking.entries) == 2
        assert ranking.entries[0].rank <= ranking.entries[1].rank

    def test_normalized_scores_bounded(self):
        ranking = compute_ranking(self.traces)
        for entry in ranking.entries:
            assert 0.0 <= entry.mean_normalized <= 1.0

    def test_subscript_access(self):
        ranking = compute_ranking(self.traces)
        assert "OptA" in ranking
        entry = ranking["OptA"]
        assert isinstance(entry.rank, float)

    def test_at_cu(self):
        ranking = compute_ranking(self.traces, at_cu=10.0)
        for entry in ranking.entries:
            assert entry.mean_normalized is not None

    def test_pvalues_exist(self):
        ranking = compute_ranking(self.traces)
        assert len(ranking.pvalues) >= 1
        for key, val in ranking.pvalues.items():
            assert len(key) == 2
            assert isinstance(val, float)

    def test_str_output(self):
        ranking = compute_ranking(self.traces)
        text = str(ranking)
        assert "Ranking" in text
        assert "OptA" in text
        assert "OptB" in text

    def test_to_dataframe(self):
        ranking = compute_ranking(self.traces)
        df = ranking.to_dataframe()
        assert "rank" in df.columns
        assert len(df) == 2

    def test_pvalues_dataframe(self):
        ranking = compute_ranking(self.traces)
        pdf = ranking.pvalues_dataframe()
        assert pdf.shape == (2, 2)

    def test_few_functions_warning(self):
        traces = {
            ("F1", "OptA", 0): _trace([1.0]),
            ("F1", "OptB", 0): _trace([2.0]),
        }
        with pytest.warns(UserWarning, match="Only 1 functions"):
            compute_ranking(traces)


class TestERTTable:
    def test_repr(self):
        table = ERTTable({}, 1.0, [], [])
        assert "ERTTable" in repr(table)

    def test_contains_false(self):
        table = ERTTable({}, 1.0, [], [])
        assert "nonexistent" not in table


class TestRankingTable:
    def test_repr(self):
        table = RankingTable([], {}, 0.05, {})
        assert "RankingTable" in repr(table)

    def test_contains_false(self):
        table = RankingTable([], {}, 0.05, {})
        assert "nonexistent" not in table
