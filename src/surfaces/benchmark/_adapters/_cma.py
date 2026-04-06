"""Adapter for pycma (CMA-ES).

The user passes cma.CMAEvolutionStrategy. The adapter derives x0
from search space bounds and buffers the population so that
ask/tell presents a single-point interface to the runner.
"""

from __future__ import annotations

from typing import Any

from surfaces.benchmark._adapters._base import AskTellAdapter, extract_bounds


class CMAAdapter(AskTellAdapter):
    def __init__(self, cls: type, params: dict):
        self._cls = cls
        self._params = dict(params)

    @property
    def name(self) -> str:
        return "cma.CMA-ES"

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

    def ask(self) -> dict[str, Any]:
        if not self._population:
            self._raw_population = self._es.ask()
            self._population = list(self._raw_population)
            self._scores = []

        x = self._population.pop(0)
        return {name: float(xi) for name, xi in zip(self._names, x)}

    def tell(self, params: dict[str, Any], score: float) -> None:
        self._scores.append(score)

        if not self._population:
            self._es.tell(self._raw_population, self._scores)


ADAPTER_CLASS = CMAAdapter
