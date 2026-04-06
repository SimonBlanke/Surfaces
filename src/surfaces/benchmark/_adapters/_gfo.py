"""Adapter for Gradient-Free-Optimizers (sealed loop).

The user passes an optimizer class from gradient_free_optimizers
(e.g. HillClimbingOptimizer). GFO maximizes by default, so the
adapter negates scores for the optimizer while recording originals.
"""

from __future__ import annotations

import time
from typing import Any

from surfaces.benchmark._adapters._base import SealedAdapter


class GFOAdapter(SealedAdapter):
    def __init__(self, cls: type, params: dict):
        self._cls = cls
        self._params = dict(params)

    @property
    def name(self) -> str:
        return self._cls.__name__

    def run(
        self,
        objective: callable,
        search_space: dict,
        seed: int,
        budget_iter: int,
    ) -> list[tuple[dict[str, Any], float, float]]:
        records: list[tuple[dict[str, Any], float, float]] = []

        def tracked(params):
            t0 = time.perf_counter()
            score = objective(params)
            elapsed = time.perf_counter() - t0
            records.append((dict(params), float(score), elapsed))
            # GFO maximizes, negate so it finds our minimum
            return -score

        import numpy as np

        np.random.seed(seed)

        params = dict(self._params)
        opt = self._cls(search_space, **params)
        opt.search(
            tracked,
            n_iter=budget_iter,
            verbosity=False,
        )

        return records


ADAPTER_CLASS = GFOAdapter
