"""Base adapter classes for optimizer integration.

Two patterns exist:

AskTellAdapter: for optimizers that support ask/tell. The benchmark
runner drives the loop and can measure overhead per step precisely.

SealedAdapter: for optimizers that run their own loop internally.
Overhead is measured by subtracting total eval time from wall time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def extract_bounds(search_space: dict) -> dict[str, tuple[float, float]]:
    """Extract (min, max) bounds from a Surfaces search space dict.

    Surfaces search spaces map parameter names to numpy arrays of
    discrete values. This extracts the continuous bounds.
    """
    return {
        name: (float(values.min()), float(values.max())) for name, values in search_space.items()
    }


class AskTellAdapter(ABC):
    """Internal adapter for optimizers with an ask/tell interface.

    Subclasses must implement setup/ask/tell and the name property.
    Population-based optimizers should buffer internally so that
    ask() always returns a single point as a param dict.
    """

    is_sealed = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique display name in the format ``package.ClassName``."""

    @abstractmethod
    def setup(self, search_space: dict, seed: int, budget: int) -> None:
        """Configure the optimizer for a specific function and run.

        Called once before each (function, seed) combination.
        The adapter must be able to handle repeated setup calls.
        """

    @abstractmethod
    def ask(self) -> dict[str, Any]:
        """Return the next point to evaluate as a param dict."""

    @abstractmethod
    def tell(self, params: dict[str, Any], score: float) -> None:
        """Report the evaluation result for the last asked point."""


class SealedAdapter(ABC):
    """Internal adapter for optimizers without ask/tell.

    These run the optimization loop internally. The adapter wraps
    the objective to track individual evaluation times. Total overhead
    is computed as wall_time - sum(eval_times) by the runner.
    """

    is_sealed = True

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique display name in the format ``package.ClassName``."""

    @abstractmethod
    def run(
        self,
        objective: callable,
        search_space: dict,
        seed: int,
        budget_iter: int,
    ) -> list[tuple[dict[str, Any], float, float]]:
        """Run the full optimization and return evaluation records.

        Returns
        -------
        list of (params, score, eval_seconds)
            Each tuple contains the parameter dict, the score returned
            by the objective, and the wall-clock seconds spent in that
            single evaluation.
        """
