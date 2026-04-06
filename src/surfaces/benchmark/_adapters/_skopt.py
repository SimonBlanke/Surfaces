"""Adapter for scikit-optimize (skopt).

The user passes skopt.Optimizer (the class). The adapter creates
Real dimensions from search space bounds and uses ask/tell.
"""

from __future__ import annotations

from typing import Any

from surfaces.benchmark._adapters._base import AskTellAdapter, extract_bounds


class SkoptAdapter(AskTellAdapter):
    def __init__(self, cls: type, params: dict):
        self._cls = cls
        self._params = dict(params)

    @property
    def name(self) -> str:
        return f"skopt.{self._cls.__name__}"

    def setup(self, search_space: dict, seed: int, budget: int) -> None:
        from skopt.space import Real

        bounds = extract_bounds(search_space)
        self._names = list(bounds.keys())
        dimensions = [Real(lo, hi, name=name) for name, (lo, hi) in bounds.items()]

        params = dict(self._params)
        params.setdefault("random_state", seed)

        self._opt = self._cls(dimensions=dimensions, **params)
        self._next_x = None

    def ask(self) -> dict[str, Any]:
        self._next_x = self._opt.ask()
        return {name: float(xi) for name, xi in zip(self._names, self._next_x)}

    def tell(self, params: dict[str, Any], score: float) -> None:
        self._opt.tell(self._next_x, score)


ADAPTER_CLASS = SkoptAdapter
