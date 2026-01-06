# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Utility functions for the collection module."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

if TYPE_CHECKING:
    from ..test_functions._base_test_function import BaseTestFunction


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

    if ".algebraic" in module:
        return "algebraic"
    elif ".bbob" in module:
        return "bbob"
    elif ".cec" in module:
        return "cec"
    elif ".engineering" in module:
        return "engineering"
    elif ".machine_learning" in module:
        return "ml"
    else:
        return "other"


def instantiate(
    functions: List[Type["BaseTestFunction"]], n_dim: int = 10
) -> List["BaseTestFunction"]:
    """Instantiate all functions in a list.

    Handles the complexity of mixed function types (fixed-dimension vs scalable).
    Fixed-dimension functions are instantiated directly, while scalable functions
    receive the n_dim parameter.

    Parameters
    ----------
    functions : list
        List of function classes.
    n_dim : int, default=10
        Number of dimensions for scalable functions.

    Returns
    -------
    list
        List of instantiated function objects.

    Examples
    --------
    >>> from surfaces import collection
    >>> functions = collection.quick.instantiate(n_dim=10)
    >>> for func in functions:
    ...     result = optimizer.minimize(func)
    """
    instances = []
    for FuncClass in functions:
        sig = inspect.signature(FuncClass.__init__)
        params = sig.parameters

        # Check if n_dim is a required parameter (no default value)
        if "n_dim" in params:
            param = params["n_dim"]
            if param.default is inspect.Parameter.empty:
                instances.append(FuncClass(n_dim=n_dim))
            else:
                instances.append(FuncClass())
        else:
            instances.append(FuncClass())

    return instances
