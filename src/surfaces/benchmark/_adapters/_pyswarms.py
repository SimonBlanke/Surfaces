"""Adapter for PySwarms (sealed loop).

The user passes a PSO class (e.g. pyswarms.single.GlobalBestPSO).
The optimizer evaluates swarms in batch, so the adapter loops
over particles to record individual eval times.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from surfaces.benchmark._adapters._base import SealedAdapter, extract_bounds


class PySwarmsAdapter(SealedAdapter):
    def __init__(self, cls: type, params: dict):
        self._cls = cls
        self._params = dict(params)

    @property
    def name(self) -> str:
        return f"pyswarms.{self._cls.__name__}"

    def run(
        self,
        objective: callable,
        search_space: dict,
        seed: int,
        budget_iter: int,
    ) -> list[tuple[dict[str, Any], float, float]]:
        bounds_dict = extract_bounds(search_space)
        names = list(bounds_dict.keys())
        n_dim = len(names)
        lb = np.array([bounds_dict[n][0] for n in names])
        ub = np.array([bounds_dict[n][1] for n in names])

        records: list[tuple[dict[str, Any], float, float]] = []

        def tracked_objective(X):
            """PySwarms vectorized objective: (n_particles, n_dim) -> (n_particles,)"""
            scores = np.empty(X.shape[0])
            for i, x in enumerate(X):
                params = {name: float(xi) for name, xi in zip(names, x)}
                t0 = time.perf_counter()
                scores[i] = objective(params)
                elapsed = time.perf_counter() - t0
                records.append((params, float(scores[i]), elapsed))
            return scores

        params = dict(self._params)
        n_particles = params.pop("n_particles", 10)
        options = params.pop("options", {"c1": 0.5, "c2": 0.3, "w": 0.9})

        np.random.seed(seed)
        opt = self._cls(
            n_particles=n_particles,
            dimensions=n_dim,
            options=options,
            bounds=(lb, ub),
            **params,
        )
        opt.optimize(tracked_objective, iters=budget_iter, verbose=False)

        return records


ADAPTER_CLASS = PySwarmsAdapter
