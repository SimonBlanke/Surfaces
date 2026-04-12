"""Tests for DataFrame export methods in benchmark statistics. Requires pandas."""

from __future__ import annotations

from surfaces.benchmark._statistics import compute_ert, compute_ranking
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


class TestERTToDataframe:
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

    def test_to_dataframe(self):
        ert = compute_ert(self.traces, self.optimal, precision=1.0)
        df = ert.to_dataframe()
        assert "ert_cu" in df.columns
        assert "success_rate" in df.columns
        assert len(df) == 2


class TestRankingToDataframe:
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

    def test_to_dataframe(self):
        ranking = compute_ranking(self.traces)
        df = ranking.to_dataframe()
        assert "rank" in df.columns
        assert len(df) == 2

    def test_pvalues_dataframe(self):
        ranking = compute_ranking(self.traces)
        pdf = ranking.pvalues_dataframe()
        assert pdf.shape == (2, 2)
