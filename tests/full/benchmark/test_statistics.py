"""Tests for compute_ranking (requires scipy)."""

from __future__ import annotations

import pytest

from surfaces.benchmark._statistics import compute_ranking
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

    def test_few_functions_warning(self):
        traces = {
            ("F1", "OptA", 0): _trace([1.0]),
            ("F1", "OptB", 0): _trace([2.0]),
        }
        with pytest.warns(UserWarning, match="Only 1 functions"):
            compute_ranking(traces)
