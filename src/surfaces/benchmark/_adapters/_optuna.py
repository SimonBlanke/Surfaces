"""Adapter for Optuna samplers.

The user passes a sampler class (e.g. optuna.samplers.TPESampler)
and optional constructor kwargs. The adapter creates a Study
internally and uses ask/tell with FloatDistribution.
"""

from __future__ import annotations

from typing import Any

from surfaces.benchmark._adapters._base import AskTellAdapter, extract_bounds


class OptunaAdapter(AskTellAdapter):
    def __init__(self, cls: type, params: dict):
        self._cls = cls
        self._params = dict(params)

    @property
    def name(self) -> str:
        return f"optuna.{self._cls.__name__}"

    def setup(self, search_space: dict, seed: int, budget: int) -> None:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = self._cls(seed=seed, **self._params)
        self._study = optuna.create_study(sampler=sampler)
        self._distributions = {
            name: optuna.distributions.FloatDistribution(lo, hi)
            for name, (lo, hi) in extract_bounds(search_space).items()
        }
        self._trial = None

    def ask(self) -> dict[str, Any]:
        self._trial = self._study.ask(fixed_distributions=self._distributions)
        return dict(self._trial.params)

    def tell(self, params: dict[str, Any], score: float) -> None:
        self._study.tell(self._trial, score)


ADAPTER_CLASS = OptunaAdapter
