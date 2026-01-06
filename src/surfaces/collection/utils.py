# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Utility functions for the collection module."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type


def get_merged_spec(func_cls: Type) -> Dict[str, Any]:
    """Get merged spec from class hierarchy."""
    result = {}
    for klass in reversed(func_cls.__mro__):
        if hasattr(klass, "_spec"):
            result.update(klass._spec)
    return result


def get_spec_value(func_cls: Type, key: str) -> Any:
    """Get a specific spec value from a function class."""
    spec = get_merged_spec(func_cls)
    return spec.get(key)


def get_n_dim(func_cls: Type) -> Optional[int]:
    """Get n_dim from spec or class attribute."""
    # Check spec first
    spec = get_merged_spec(func_cls)
    if "n_dim" in spec and spec["n_dim"] is not None:
        return spec["n_dim"]

    # Check class attribute (for fixed-dimension functions)
    if hasattr(func_cls, "n_dim") and func_cls.n_dim is not None:
        return func_cls.n_dim

    return None


def get_category(func_cls: Type) -> str:
    """Determine the category of a function class."""
    module = func_cls.__module__

    # Check more specific patterns first
    if ".constrained" in module:
        return "constrained"
    elif ".algebraic" in module:
        return "algebraic"
    elif ".bbob" in module:
        return "bbob"
    elif ".cec" in module:
        return "cec"
    elif ".machine_learning" in module:
        return "ml"
    elif ".simulation" in module:
        return "simulation"
    else:
        return "other"
