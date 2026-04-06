"""Adapter for SMAC3.

The user passes a facade class (e.g. smac.HyperparameterOptimizationFacade).
The adapter builds ConfigSpace and Scenario from search space bounds.
"""

from __future__ import annotations

from typing import Any

from surfaces._benchmark._adapters._base import AskTellAdapter, extract_bounds


class SMACAdapter(AskTellAdapter):
    def __init__(self, cls: type, params: dict):
        self._cls = cls
        self._params = dict(params)

    @property
    def name(self) -> str:
        return "SMAC"

    def setup(self, search_space: dict, seed: int, budget: int) -> None:
        from ConfigSpace import ConfigurationSpace, Float
        from smac import Scenario

        bounds = extract_bounds(search_space)
        cs = ConfigurationSpace(seed=seed)
        for param_name, (lo, hi) in bounds.items():
            cs.add(Float(param_name, (lo, hi)))

        scenario = Scenario(
            cs,
            deterministic=True,
            n_trials=budget,
            seed=seed,
        )

        # SMAC requires a target function at init even in ask/tell mode
        def _dummy(config, seed=0):
            return 0.0

        params = dict(self._params)
        params.setdefault("overwrite", True)

        self._smac = self._cls(scenario, _dummy, **params)
        self._info = None
        self._best_params: dict | None = None
        self._best_score = float("inf")

    def ask(self) -> dict[str, Any]:
        self._info = self._smac.ask()
        return dict(self._info.config)

    def tell(self, params: dict[str, Any], score: float) -> None:
        from smac.runhistory.dataclasses import TrialValue

        value = TrialValue(cost=score)
        self._smac.tell(self._info, value)
        if score < self._best_score:
            self._best_score = score
            self._best_params = dict(params)

    def best(self) -> tuple[dict[str, Any], float]:
        if self._best_params is None:
            raise RuntimeError("No evaluations recorded yet")
        return self._best_params, self._best_score


ADAPTER_CLASS = SMACAdapter
