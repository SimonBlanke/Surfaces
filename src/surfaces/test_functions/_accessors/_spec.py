"""SpecAccessor: dict-like access to function characteristics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np

    from .._base_test_function import BaseTestFunction


class SpecAccessor:
    """Namespaced access to function specification/characteristics.

    Wraps the MRO-based _spec merging and provides dict-like access
    plus typed properties for common spec fields.

    Parameters
    ----------
    func : BaseTestFunction
        The test function instance.
    """

    def __init__(self, func: "BaseTestFunction") -> None:
        self._func = func

    def as_dict(self) -> dict:
        """Return the full merged spec as a plain dict."""
        result = {}
        for klass in reversed(type(self._func).__mro__):
            if hasattr(klass, "_spec"):
                result.update(klass._spec)
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-compatible get."""
        return self.as_dict().get(key, default)

    # Dict-like protocol

    def __getitem__(self, key: str) -> Any:
        d = self.as_dict()
        if key not in d:
            raise KeyError(key)
        return d[key]

    def __contains__(self, key: str) -> bool:
        return key in self.as_dict()

    def __repr__(self) -> str:
        return f"SpecAccessor({self.as_dict()!r})"

    # Typed properties from _spec

    @property
    def convex(self) -> bool:
        return self.as_dict().get("convex", False)

    @property
    def unimodal(self) -> bool:
        return self.as_dict().get("unimodal", False)

    @property
    def separable(self) -> bool:
        return self.as_dict().get("separable", False)

    @property
    def continuous(self) -> bool:
        return self.as_dict().get("continuous", True)

    @property
    def differentiable(self) -> bool:
        return self.as_dict().get("differentiable", True)

    @property
    def scalable(self) -> bool:
        return self.as_dict().get("scalable", False)

    @property
    def n_objectives(self) -> int:
        return self.as_dict().get("n_objectives", 1)

    @property
    def default_bounds(self) -> tuple:
        return self.as_dict().get("default_bounds", (-5.0, 5.0))

    # Global optimum (from class-level attrs on concrete functions)

    @property
    def f_global(self) -> Optional[float]:
        return getattr(self._func, "f_global", None)

    @property
    def x_global(self) -> "Optional[np.ndarray]":
        return getattr(self._func, "x_global", None)
