"""Adapter for pycma (CMA-ES).

The user passes cma.CMAEvolutionStrategy. The adapter derives x0
from search space bounds and buffers the population so that
ask/tell presents a single-point interface to the runner.
"""

from __future__ import annotations

from typing import Any

from surfaces._benchmark._adapters._base import AskTellAdapter, extract_bounds


class CMAAdapter(AskTellAdapter):
    def __init__(self, cls: type, params: dict):
        self._cls = cls
        self._params = dict(params)

    @property
    def name(self) -> str:
        return "CMA-ES"

    def setup(self, search_space: dict, seed: int, budget: int) -> None:
        bounds = extract_bounds(search_space)
        self._names = list(bounds.keys())
        lowers = [bounds[n][0] for n in self._names]
        uppers = [bounds[n][1] for n in self._names]

        x0 = [(lo + hi) / 2 for lo, hi in zip(lowers, uppers)]

        params = dict(self._params)
        sigma0 = params.pop("sigma0", (uppers[0] - lowers[0]) / 4)

        opts = {"bounds": [lowers, uppers], "seed": seed}
        opts.update(params)

        self._es = self._cls(x0, sigma0, opts)
        self._population: list = []
        self._raw_population: list = []
        self._scores: list[float] = []
        self._best_params: dict | None = None
        self._best_score = float("inf")

    def ask(self) -> dict[str, Any]:
        if not self._population:
            self._raw_population = self._es.ask()
            self._population = list(self._raw_population)
            self._scores = []

        x = self._population.pop(0)
        return {name: float(xi) for name, xi in zip(self._names, x)}

    def tell(self, params: dict[str, Any], score: float) -> None:
        self._scores.append(score)
        if score < self._best_score:
            self._best_score = score
            self._best_params = dict(params)

        # Feed the full generation back once all members are evaluated
        if not self._population:
            self._es.tell(self._raw_population, self._scores)

    def best(self) -> tuple[dict[str, Any], float]:
        if self._best_params is None:
            raise RuntimeError("No evaluations recorded yet")
        return self._best_params, self._best_score


ADAPTER_CLASS = CMAAdapter
