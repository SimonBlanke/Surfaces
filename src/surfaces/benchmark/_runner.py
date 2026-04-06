"""Benchmark runner: orchestrates function x optimizer x seed runs.

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

from surfaces._benchmark._resolve import resolve_functions, resolve_optimizer
from surfaces._benchmark._result import BenchmarkResult
from surfaces._benchmark._suites import Suite
from surfaces._benchmark._trace import EvalRecord, Trace
from surfaces._cost import calibrate, to_cu


def run(
    functions: Any,
    optimizers: list,
    budget_cu: float | None = None,
    budget_iter: int | None = None,
    n_seeds: int = 1,
    seed: int = 0,
) -> BenchmarkResult:
    """Run a benchmark across functions, optimizers, and seeds.

    Parameters
    ----------
    functions
        Test functions to benchmark on. Accepts a single class,
        a list of classes, or a Collection.
    optimizers
        Optimizer specs. Each element can be:
        - A class (auto-detected by module path)
        - A (class, params_dict) tuple
        - An instance with ask() and tell() methods
    budget_cu : float, optional
        Maximum compute budget per run in Compute Units.
    budget_iter : int, optional
        Maximum number of function evaluations per run.
    n_seeds : int
        Number of independent runs per (function, optimizer) pair.
    seed : int
        Base random seed. Run i uses seed + i.

    Returns
    -------
    BenchmarkResult
    """
    if budget_cu is None and budget_iter is None:
        raise ValueError("Specify at least one of budget_cu or budget_iter")

    calibrate()

    func_classes = resolve_functions(functions)
    adapters = [resolve_optimizer(spec) for spec in optimizers]

    traces: dict[tuple[str, str, int], Trace] = {}

    for func_cls in func_classes:
        for adapter in adapters:
            for i in range(n_seeds):
                run_seed = seed + i
                func = _instantiate_function(func_cls)

                if adapter.is_sealed:
                    trace = _run_sealed(func, adapter, run_seed, budget_cu, budget_iter)
                else:
                    trace = _run_ask_tell(func, adapter, run_seed, budget_cu, budget_iter)

                key = (func_cls.__name__, adapter.name, run_seed)
                traces[key] = trace

    return BenchmarkResult(traces)


def run_suite(
    suite: Suite,
    optimizers: list,
    **overrides: Any,
) -> BenchmarkResult:
    """Run a pre-defined benchmark suite.

    Parameters
    ----------
    suite : Suite
        A predefined suite from surfaces._benchmark.suites.
    optimizers : list
        Optimizer specs (same format as run()).
    **overrides
        Override suite defaults (budget_cu, budget_iter, n_seeds, seed).
    """
    from surfaces import collection

    functions = collection.filter(**suite.function_filter)

    kwargs: dict[str, Any] = {
        "budget_cu": suite.budget_cu,
        "budget_iter": suite.budget_iter,
        "n_seeds": suite.n_seeds,
    }
    kwargs.update(overrides)

    return run(functions=functions, optimizers=optimizers, **kwargs)


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
    effective_budget = budget_iter or 0
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
