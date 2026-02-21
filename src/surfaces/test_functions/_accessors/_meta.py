"""MetaAccessor: read-only proxy to class-level metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .._base_test_function import BaseTestFunction


class MetaAccessor:
    """Read-only proxy to class-level metadata attributes.

    Parameters
    ----------
    func : BaseTestFunction
        The test function instance.
    """

    def __init__(self, func: "BaseTestFunction") -> None:
        self._func = func

    @property
    def name(self) -> Optional[str]:
        return getattr(self._func, "name", None)

    @property
    def latex_formula(self) -> Optional[str]:
        return getattr(self._func, "latex_formula", None)

    @property
    def reference(self) -> Optional[str]:
        return getattr(self._func, "reference", None)

    @property
    def reference_url(self) -> Optional[str]:
        return getattr(self._func, "reference_url", None)

    @property
    def tagline(self) -> Optional[str]:
        return getattr(self._func, "tagline", None)

    @property
    def func_id(self) -> Optional[str]:
        return getattr(self._func, "func_id", None)

    def __repr__(self) -> str:
        return f"MetaAccessor(name={self.name!r})"
