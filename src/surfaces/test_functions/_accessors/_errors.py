"""ErrorAccessor: Mapping-like access to error handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .._base_test_function import BaseTestFunction


class ErrorAccessor:
    """Namespaced access to error handler management.

    Mapping-like protocol (not Sequence, since it's key-value).

    Parameters
    ----------
    func : BaseTestFunction
        The test function instance.
    """

    def __init__(self, func: "BaseTestFunction") -> None:
        self._func = func

    def add(self, exception_type, return_value: float) -> None:
        """Add an error handler mapping."""
        if self._func._error_handlers is None:
            self._func._error_handlers = {}
        self._func._error_handlers[exception_type] = return_value

    def remove(self, exception_type) -> None:
        """Remove an error handler.

        Raises KeyError if the exception type is not found.
        """
        if self._func._error_handlers is None:
            raise KeyError(exception_type)
        del self._func._error_handlers[exception_type]

    def clear(self) -> None:
        """Remove all error handlers."""
        self._func._error_handlers = None

    def __len__(self) -> int:
        if self._func._error_handlers is None:
            return 0
        return len(self._func._error_handlers)

    def __contains__(self, exception_type) -> bool:
        if self._func._error_handlers is None:
            return False
        return exception_type in self._func._error_handlers

    def __iter__(self) -> Iterator:
        if self._func._error_handlers is None:
            return iter([])
        return iter(self._func._error_handlers)

    def __getitem__(self, exception_type) -> float:
        if self._func._error_handlers is None:
            raise KeyError(exception_type)
        return self._func._error_handlers[exception_type]

    def __repr__(self) -> str:
        if self._func._error_handlers is None:
            return "ErrorAccessor(disabled)"
        return f"ErrorAccessor({len(self)} handlers)"
