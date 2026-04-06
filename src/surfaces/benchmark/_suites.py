"""Pre-defined benchmark suite configurations.

Each suite specifies a set of function filter criteria and
benchmark parameters tuned for a specific analysis goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Suite:
    """A pre-configured benchmark scenario.

    Use with run_suite() or unpack manually into run().
    """

    name: str
    description: str
    function_filter: dict[str, Any]
    budget_cu: float | None = None
    budget_iter: int | None = None
    n_seeds: int = 5


overhead_analysis = Suite(
    name="overhead_analysis",
    description=(
        "Cheap algebraic functions (< 2 CU) where optimizer overhead "
        "dominates. Reveals which optimizers are lean vs. heavy."
    ),
    function_filter={
        "eval_cost": lambda c: c is not None and c < 2,
    },
    budget_cu=10_000,
    n_seeds=5,
)

expensive_functions = Suite(
    name="expensive_functions",
    description=(
        "Simulation and ML functions (> 1000 CU) where optimizer "
        "overhead is negligible. Tests pure optimization quality."
    ),
    function_filter={
        "eval_cost": lambda c: c is not None and c > 1000,
    },
    budget_cu=5_000_000,
    n_seeds=3,
)

bbob_standard = Suite(
    name="bbob_standard",
    description=(
        "All 24 BBOB functions at default dimension. The standard "
        "academic benchmark for numerical optimization."
    ),
    function_filter={
        "category": "bbob",
    },
    budget_cu=50_000,
    n_seeds=5,
)

quick = Suite(
    name="quick",
    description=("A small, fast suite for testing and CI. " "Few functions with a tight budget."),
    function_filter={
        "category": "algebraic",
        "n_dim": 2,
        "unimodal": False,
    },
    budget_cu=5_000,
    n_seeds=2,
)

ALL_SUITES: dict[str, Suite] = {
    "overhead_analysis": overhead_analysis,
    "expensive_functions": expensive_functions,
    "bbob_standard": bbob_standard,
    "quick": quick,
}
