"""Evaluation trace recording for benchmark runs."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvalRecord:
    """Single evaluation record within a benchmark trace.

    Captures both the evaluation result and the computational cost
    breakdown between function evaluation and optimizer overhead.
    """

    params: dict[str, Any]
    score: float
    eval_cu: float
    overhead_cu: float
    cumulative_cu: float
    best_so_far: float
    wall_seconds: float


class Trace:
    """Ordered sequence of evaluation records from a single benchmark run.

    Represents one complete optimization trajectory: a single optimizer
    on a single function with a single seed. The cumulative CU values
    form a monotonically increasing sequence that enables CU-indexed
    lookups via binary search.
    """

    __slots__ = ("_records", "_cu_breakpoints")

    def __init__(self):
        self._records: list[EvalRecord] = []
        self._cu_breakpoints: list[float] = []

    def append(self, record: EvalRecord) -> None:
        self._records.append(record)
        self._cu_breakpoints.append(record.cumulative_cu)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def __getitem__(self, idx):
        return self._records[idx]

    def __bool__(self) -> bool:
        return len(self._records) > 0

    @property
    def records(self) -> list[EvalRecord]:
        return list(self._records)

    @property
    def best_score(self) -> float | None:
        if not self._records:
            return None
        return self._records[-1].best_so_far

    @property
    def total_cu(self) -> float:
        if not self._records:
            return 0.0
        return self._records[-1].cumulative_cu

    @property
    def n_evaluations(self) -> int:
        return len(self._records)

    @property
    def total_overhead_cu(self) -> float:
        return sum(r.overhead_cu for r in self._records)

    @property
    def total_eval_cu(self) -> float:
        return sum(r.eval_cu for r in self._records)

    @property
    def overhead_fraction(self) -> float:
        """Fraction of total CU spent on optimizer overhead."""
        total = self.total_cu
        if total == 0:
            return 0.0
        return self.total_overhead_cu / total

    def score_at_cu(self, cu_budget: float) -> float | None:
        """Best score achieved within the given CU budget.

        Uses binary search on cumulative CU for efficient lookup.
        """
        if not self._records:
            return None
        idx = bisect.bisect_right(self._cu_breakpoints, cu_budget)
        if idx == 0:
            return None
        return self._records[idx - 1].best_so_far

    def score_at_iter(self, n_iter: int) -> float | None:
        """Best score after exactly n_iter evaluations."""
        if not self._records or n_iter < 1:
            return None
        idx = min(n_iter, len(self._records))
        return self._records[idx - 1].best_so_far
