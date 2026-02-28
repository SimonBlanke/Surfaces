"""CallbackAccessor: Sequence-like access to callbacks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator

if TYPE_CHECKING:
    from .._base_test_function import BaseTestFunction


class CallbackAccessor(Sequence):
    """Namespaced access to callback management.

    Implements the Sequence protocol for iteration and indexing.

    Parameters
    ----------
    func : BaseTestFunction
        The test function instance.
    """

    def __init__(self, func: "BaseTestFunction") -> None:
        self._func = func

    def add(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback to be invoked after each evaluation."""
        self._func._callbacks.append(callback)

    def remove(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a previously added callback.

        Raises ValueError if the callback is not found.
        """
        self._func._callbacks.remove(callback)

    def clear(self) -> None:
        """Remove all callbacks."""
        self._func._callbacks.clear()

    def __getitem__(self, index):
        return self._func._callbacks[index]

    def __len__(self) -> int:
        return len(self._func._callbacks)

    def __iter__(self) -> Iterator:
        return iter(self._func._callbacks)

    def __repr__(self) -> str:
        return f"CallbackAccessor({len(self)} callbacks)"
