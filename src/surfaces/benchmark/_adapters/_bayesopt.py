"""Adapter for the bayesian-optimization package.

The user passes bayes_opt.BayesianOptimization (the class). The
adapter uses the suggest/register interface. Note that BayesOpt
maximizes, so scores are negated internally.
"""

from __future__ import annotations

from typing import Any

from surfaces.benchmark._adapters._base import AskTellAdapter, extract_bounds


class BayesOptAdapter(AskTellAdapter):
    def __init__(self, cls: type, params: dict):
        self._cls = cls
        self._params = dict(params)

    @property
    def name(self) -> str:
        return "BayesianOptimization"

    def setup(self, search_space: dict, seed: int, budget: int) -> None:
        bounds = extract_bounds(search_space)

        params = dict(self._params)
        params.setdefault("random_state", seed)
        params.setdefault("verbose", 0)

        self._opt = self._cls(f=None, pbounds=bounds, **params)
        self._best_params: dict | None = None
        self._best_score = float("inf")

    def ask(self) -> dict[str, Any]:
        return dict(self._opt.suggest())

    def tell(self, params: dict[str, Any], score: float) -> None:
        # BayesOpt maximizes, negate so it finds our minimum
        self._opt.register(params=params, target=-score)
        if score < self._best_score:
            self._best_score = score
            self._best_params = dict(params)

    def best(self) -> tuple[dict[str, Any], float]:
        if self._best_params is None:
            raise RuntimeError("No evaluations recorded yet")
        return self._best_params, self._best_score


ADAPTER_CLASS = BayesOptAdapter
