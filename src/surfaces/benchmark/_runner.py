"""Internal benchmark execution: ask/tell and sealed run loops.

The runner handles two execution modes:

Ask/Tell adapters: the runner drives the loop, timing ask(), eval,
and tell() separately for precise overhead attribution.

Sealed adapters: the adapter runs its own loop. The runner measures
total time and subtracts recorded eval times to estimate overhead.
"""

from __future__ import annotations

import inspect
import time
from typing import Any

from surfaces._cost import to_cu
from surfaces.benchmark._trace import EvalRecord, Trace


def _instantiate_function(func_cls: type) -> Any:
    """Create a function instance configured for benchmarking."""
    sig = inspect.signature(func_cls.__init__)
    kwargs: dict[str, Any] = {}
    if "collect_data" in sig.parameters:
        kwargs["collect_data"] = False
    return func_cls(**kwargs)


def _run_ask_tell(
    func: Any,
    adapter: Any,
    seed: int,
    budget_cu: float | None,
    budget_iter: int | None,
) -> Trace:
    """Run a single benchmark with an ask/tell optimizer."""
    effective_budget = budget_iter if budget_iter is not None else 10_000
    adapter.setup(func.search_space, seed, effective_budget)

    trace = Trace()
    cumulative_cu = 0.0
    best_score = float("inf")
    iteration = 0

    while True:
        if budget_iter is not None and iteration >= budget_iter:
            break
        if budget_cu is not None and cumulative_cu >= budget_cu:
            break

        t0 = time.perf_counter()
        params = adapter.ask()
        t_ask = time.perf_counter() - t0

        t0 = time.perf_counter()
        score = float(func(params))
        t_eval = time.perf_counter() - t0

        t0 = time.perf_counter()
        adapter.tell(params, score)
        t_tell = time.perf_counter() - t0

        overhead_cu = to_cu(t_ask + t_tell)
        eval_cu = to_cu(t_eval)
        cumulative_cu += overhead_cu + eval_cu
        best_score = min(best_score, score)
        wall = t_ask + t_eval + t_tell

        trace.append(
            EvalRecord(
                params=params,
                score=score,
                eval_cu=eval_cu,
                overhead_cu=overhead_cu,
                cumulative_cu=cumulative_cu,
                best_so_far=best_score,
                wall_seconds=wall,
            )
        )

        iteration += 1

    return trace


def _run_sealed(
    func: Any,
    adapter: Any,
    seed: int,
    budget_cu: float | None,
    budget_iter: int | None,
) -> Trace:
    """Run a single benchmark with a sealed-loop optimizer.

    Overhead is estimated by subtracting total eval time from total
    wall time, then distributed evenly across evaluations.
    """
    effective_budget = budget_iter if budget_iter is not None else 10_000

    t_start = time.perf_counter()
    records = adapter.run(func, func.search_space, seed, effective_budget)
    t_total = time.perf_counter() - t_start

    if not records:
        return Trace()

    total_eval_time = sum(r[2] for r in records)
    total_overhead = max(0.0, t_total - total_eval_time)
    per_eval_overhead = total_overhead / len(records)

    trace = Trace()
    cumulative_cu = 0.0
    best_score = float("inf")

    for params, score, eval_seconds in records:
        eval_cu = to_cu(eval_seconds)
        overhead_cu = to_cu(per_eval_overhead)
        cumulative_cu += eval_cu + overhead_cu
        best_score = min(best_score, score)

        if budget_cu is not None and cumulative_cu > budget_cu:
            break

        trace.append(
            EvalRecord(
                params=params,
                score=score,
                eval_cu=eval_cu,
                overhead_cu=overhead_cu,
                cumulative_cu=cumulative_cu,
                best_so_far=best_score,
                wall_seconds=eval_seconds + per_eval_overhead,
            )
        )

    return trace
