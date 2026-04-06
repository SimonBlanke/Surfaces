"""CU-based benchmark module for gradient-free optimizer comparison.

Provides hardware-independent benchmarking using Compute Units (CU)
as the cost metric. Optimizers are auto-detected via duck typing;
pass any known optimizer class and the module handles the rest.

Usage::

    from surfaces._benchmark import run
    from surfaces import collection

    result = run(
        functions=collection.filter(eval_cost=lambda c: c is not None and c < 10),
        optimizers=[SomeOptimizerClass, (AnotherClass, {"param": 1})],
        budget_cu=50_000,
        n_seeds=5,
    )
    print(result.summary())
"""

from surfaces._benchmark._result import BenchmarkResult
from surfaces._benchmark._runner import run, run_suite
from surfaces._benchmark._suites import ALL_SUITES as suites
from surfaces._benchmark._suites import Suite
from surfaces._benchmark._trace import EvalRecord, Trace

__all__ = [
    "run",
    "run_suite",
    "BenchmarkResult",
    "EvalRecord",
    "Trace",
    "Suite",
    "suites",
]
