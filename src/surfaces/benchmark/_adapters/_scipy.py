"""Adapter for scipy.optimize functions (sealed loop).

The user passes a scipy optimizer function like
scipy.optimize.differential_evolution. The adapter wraps the
objective to track individual eval times and computes overhead
by subtraction.
"""

from __future__ import annotations

import inspect
import time
from typing import Any

from surfaces.benchmark._adapters._base import SealedAdapter, extract_bounds


class ScipyAdapter(SealedAdapter):
    def __init__(self, func: callable, params: dict):
        self._scipy_func = func
        self._params = dict(params)

    @property
    def name(self) -> str:
        return f"scipy.{self._scipy_func.__name__}"

    def run(
        self,
        objective: callable,
        search_space: dict,
        seed: int,
        budget_iter: int,
    ) -> list[tuple[dict[str, Any], float, float]]:
        bounds_dict = extract_bounds(search_space)
        names = list(bounds_dict.keys())
        bounds = [bounds_dict[n] for n in names]

        records: list[tuple[dict[str, Any], float, float]] = []

        def tracked(x):
            params = {name: float(xi) for name, xi in zip(names, x)}
            t0 = time.perf_counter()
            score = objective(params)
            elapsed = time.perf_counter() - t0
            records.append((params, float(score), elapsed))
            return float(score)

        kwargs = dict(self._params)

        # Set defaults based on what the scipy function accepts
        sig = inspect.signature(self._scipy_func)
        if "seed" in sig.parameters:
            kwargs.setdefault("seed", seed)
        if "maxiter" in sig.parameters:
            kwargs.setdefault("maxiter", budget_iter)
        # polish runs an extra local optimizer at the end, skip for clean benchmarking
        if "polish" in sig.parameters:
            kwargs.setdefault("polish", False)

        self._scipy_func(tracked, bounds, **kwargs)
        return records


ADAPTER_CLASS = ScipyAdapter
