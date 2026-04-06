"""Optimizer and function resolution via duck typing.

The resolver inspects an optimizer object's module path to identify
which package it belongs to, then lazily loads the appropriate
internal adapter. For unknown optimizers with ask/tell methods,
a generic adapter is used as fallback.
"""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE_REGISTRY: dict[str, str] = {
    "optuna": "_optuna",
    "cma": "_cma",
    "scipy": "_scipy",
    "nevergrad": "_nevergrad",
    "bayes_opt": "_bayesopt",
    "pymoo": "_pymoo",
    "skopt": "_skopt",
    "smac": "_smac",
    "pyswarms": "_pyswarms",
    "gradient_free_optimizers": "_gfo",
}


def resolve_optimizer(spec: Any) -> Any:
    """Resolve an optimizer spec into an internal adapter.

    Accepts:
        SomeClass                      - optimizer class
        (SomeClass, {"param": value})  - class with config
        some_instance                  - object with ask/tell methods
    """
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise TypeError(
                f"Optimizer tuple must be (class, params_dict), got {len(spec)} elements"
            )
        obj, params = spec
        if not isinstance(params, dict):
            raise TypeError(
                f"Second element of optimizer tuple must be a dict, got {type(params).__name__}"
            )
    else:
        obj, params = spec, {}

    module = _get_module(obj)

    for prefix, adapter_module_name in _PACKAGE_REGISTRY.items():
        if module.startswith(prefix):
            return _load_adapter(adapter_module_name, obj, params)

    if _has_ask_tell(obj):
        from surfaces.benchmark._adapters._generic import GenericAskTellAdapter

        return GenericAskTellAdapter(obj, params)

    raise TypeError(
        f"Cannot resolve optimizer: {obj!r} (module: {module}). "
        f"Pass a known optimizer class or an object with ask() and tell() methods."
    )


def resolve_functions(functions: Any) -> list:
    """Resolve a function spec into a list of function classes.

    Accepts a single class, a list of classes, or a Collection.
    """
    if isinstance(functions, type):
        return [functions]
    return list(functions)


def _get_module(obj: Any) -> str:
    """Get the module path of a class, instance, or callable."""
    if isinstance(obj, type):
        return obj.__module__ or ""
    if callable(obj):
        return getattr(obj, "__module__", "") or ""
    return type(obj).__module__ or ""


def _load_adapter(adapter_module_name: str, obj: Any, params: dict) -> Any:
    """Lazily import an adapter module and instantiate it."""
    mod = importlib.import_module(f"surfaces.benchmark._adapters.{adapter_module_name}")
    return mod.ADAPTER_CLASS(obj, params)


def _has_ask_tell(obj: Any) -> bool:
    """Check if an object or class has ask and tell methods."""
    target = obj if isinstance(obj, type) else type(obj)
    return callable(getattr(target, "ask", None)) and callable(getattr(target, "tell", None))
