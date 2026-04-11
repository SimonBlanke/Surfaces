"""Benchmark module for gradient-free optimizer comparison.

Provides hardware-independent benchmarking using Compute Units (CU)
as the cost metric. Optimizers are auto-detected via duck typing;
pass any known optimizer class and the module handles the rest.

Usage::

    from surfaces.benchmark import Benchmark
    from surfaces import collection

    bench = Benchmark(budget_cu=50_000, n_seeds=5)
    bench.add_functions(collection.filter(category="bbob"))
    bench.add_optimizers([SomeOptimizerClass, (AnotherClass, {"param": 1})])
    bench.run()
    print(bench.results.summary())
"""

from surfaces.benchmark._backends import ParallelBackend, ProcessBackend, ThreadBackend
from surfaces.benchmark._benchmark import Benchmark
from surfaces.benchmark._progress import TrialInfo
from surfaces.benchmark._suites import ALL_SUITES as suites
from surfaces.benchmark._suites import Suite
from surfaces.benchmark._trace import EvalRecord, Trace

__all__ = [
    "Benchmark",
    "EvalRecord",
    "ParallelBackend",
    "ProcessBackend",
    "Suite",
    "ThreadBackend",
    "Trace",
    "TrialInfo",
    "suites",
]
