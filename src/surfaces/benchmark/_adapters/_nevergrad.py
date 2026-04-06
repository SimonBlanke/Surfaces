"""Adapter for Nevergrad optimizers.

The user passes an optimizer class from nevergrad.optimizers
(e.g. ng.optimizers.OnePlusOne, ng.optimizers.NGOpt). The adapter
builds the parametrization from search space bounds.
"""

from __future__ import annotations

from typing import Any

from surfaces.benchmark._adapters._base import AskTellAdapter, extract_bounds


class NevergradAdapter(AskTellAdapter):
    def __init__(self, cls: type, params: dict):
        self._cls = cls
        self._params = dict(params)

    @property
    def name(self) -> str:
        return self._cls.__name__

    def setup(self, search_space: dict, seed: int, budget: int) -> None:
        import nevergrad as ng
        import numpy as np

        bounds = extract_bounds(search_space)
        self._names = sorted(bounds.keys())
        n_dim = len(self._names)
        lowers = [bounds[n][0] for n in self._names]
        uppers = [bounds[n][1] for n in self._names]

        parametrization = ng.p.Array(
            shape=(n_dim,),
            lower=lowers,
            upper=uppers,
        )
        parametrization.random_state = np.random.RandomState(seed)

        self._opt = self._cls(
            parametrization=parametrization,
            budget=budget,
            **self._params,
        )
        self._candidate = None
        self._best_params: dict | None = None
        self._best_score = float("inf")

    def ask(self) -> dict[str, Any]:
        self._candidate = self._opt.ask()
        x = self._candidate.value
        return {name: float(xi) for name, xi in zip(self._names, x)}

    def tell(self, params: dict[str, Any], score: float) -> None:
        self._opt.tell(self._candidate, score)
        if score < self._best_score:
            self._best_score = score
            self._best_params = dict(params)

    def best(self) -> tuple[dict[str, Any], float]:
        if self._best_params is None:
            raise RuntimeError("No evaluations recorded yet")
        return self._best_params, self._best_score


ADAPTER_CLASS = NevergradAdapter
