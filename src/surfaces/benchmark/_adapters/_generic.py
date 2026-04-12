"""Generic adapter for custom optimizers with ask/tell methods.

This is the fallback adapter for optimizers that don't match any
known package. The optimizer must either be a class whose constructor
accepts (search_space, seed=...) or an instance with an optional
setup(search_space, seed=...) method.
"""

from __future__ import annotations

from typing import Any

from surfaces.benchmark._adapters._base import AskTellAdapter


class GenericAskTellAdapter(AskTellAdapter):
    def __init__(self, cls_or_instance: Any, params: dict):
        self._obj = cls_or_instance
        self._params = dict(params)
        self._is_class = isinstance(cls_or_instance, type)
        self._opt = None

    @property
    def name(self) -> str:
        cls = self._obj if self._is_class else type(self._obj)
        module = (cls.__module__ or "").split(".")[0]
        return f"{module}.{cls.__name__}"

    def setup(self, search_space: dict, seed: int, budget: int) -> None:
        if self._is_class:
            self._opt = self._obj(search_space, seed=seed, **self._params)
        else:
            self._opt = self._obj
            if hasattr(self._opt, "setup"):
                self._opt.setup(search_space, seed=seed)

    def ask(self) -> dict[str, Any]:
        return self._opt.ask()

    def tell(self, params: dict[str, Any], score: float) -> None:
        self._opt.tell(params, score)


ADAPTER_CLASS = GenericAskTellAdapter
