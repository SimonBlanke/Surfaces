"""MemoryAccessor: cache management."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._base_test_function import BaseTestFunction


class MemoryAccessor:
    """Namespaced access to memory cache management.

    Parameters
    ----------
    func : BaseTestFunction
        The test function instance.
    """

    def __init__(self, func: "BaseTestFunction") -> None:
        self._func = func

    @property
    def enabled(self) -> bool:
        """Whether memory caching is enabled."""
        return self._func._memory_enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._func._memory_enabled = value

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._func._memory_cache)

    def reset(self) -> None:
        """Clear the memory cache."""
        self._func._memory_cache.clear()

    def __repr__(self) -> str:
        return f"MemoryAccessor(enabled={self.enabled}, size={self.size})"
