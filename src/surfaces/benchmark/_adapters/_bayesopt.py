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
        return "bayes_opt.BayesianOptimization"

    def setup(self, search_space: dict, seed: int, budget: int) -> None:
        bounds = extract_bounds(search_space)

        params = dict(self._params)
        params.setdefault("random_state", seed)
        params.setdefault("verbose", 0)

        self._opt = self._cls(f=None, pbounds=bounds, **params)

    def ask(self) -> dict[str, Any]:
        return dict(self._opt.suggest())

    def tell(self, params: dict[str, Any], score: float) -> None:
        self._opt.register(params=params, target=-score)


ADAPTER_CLASS = BayesOptAdapter
