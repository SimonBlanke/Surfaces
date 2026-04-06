"""Adapter for pymoo algorithms.

The user passes an algorithm class (e.g. pymoo.algorithms.soo.nonconvex.ga.GA
or pymoo.algorithms.moo.nsga2.NSGA2). Population-based ask/tell is
buffered so the runner sees a single-point interface.
"""

from __future__ import annotations

from typing import Any

from surfaces._benchmark._adapters._base import AskTellAdapter, extract_bounds


class PymooAdapter(AskTellAdapter):
    def __init__(self, cls: type, params: dict):
        self._cls = cls
        self._params = dict(params)

    @property
    def name(self) -> str:
        return self._cls.__name__

    def setup(self, search_space: dict, seed: int, budget: int) -> None:
        import numpy as np
        from pymoo.core.problem import Problem

        bounds = extract_bounds(search_space)
        self._names = list(bounds.keys())
        n_dim = len(self._names)
        xl = np.array([bounds[n][0] for n in self._names])
        xu = np.array([bounds[n][1] for n in self._names])

        problem = Problem(n_var=n_dim, n_obj=1, xl=xl, xu=xu)
        self._algorithm = self._cls(**self._params)
        self._algorithm.setup(problem, termination=("n_eval", budget), seed=seed)

        self._population: list = []
        self._raw_pop = None
        self._scores: list[list[float]] = []
        self._best_params: dict | None = None
        self._best_score = float("inf")

    def ask(self) -> dict[str, Any]:
        if not self._population:
            self._raw_pop = self._algorithm.ask()
            X = self._raw_pop.get("X")
            self._population = list(X)
            self._scores = []

        x = self._population.pop(0)
        return {name: float(xi) for name, xi in zip(self._names, x)}

    def tell(self, params: dict[str, Any], score: float) -> None:
        self._scores.append([score])
        if score < self._best_score:
            self._best_score = score
            self._best_params = dict(params)

        if not self._population:
            import numpy as np

            self._raw_pop.set("F", np.array(self._scores))
            self._algorithm.tell(infills=self._raw_pop)

    def best(self) -> tuple[dict[str, Any], float]:
        if self._best_params is None:
            raise RuntimeError("No evaluations recorded yet")
        return self._best_params, self._best_score


ADAPTER_CLASS = PymooAdapter
