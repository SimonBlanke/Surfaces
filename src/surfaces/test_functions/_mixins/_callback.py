# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Callback management mixin for test functions."""

from typing import Any, Callable, Dict, List


class CallbackMixin:
    """Mixin providing callback management for test functions.

    Callbacks are invoked after each evaluation with a record dict containing
    all parameters and the score. Use callbacks for logging, streaming to
    external systems, or custom processing.

    Attributes
    ----------
    _callbacks : list of callables
        Internal list of registered callbacks.
    """

    _callbacks: List[Callable[[Dict[str, Any]], None]]

    def _init_callbacks(self, callbacks=None) -> None:
        """Initialize callback list.

        Parameters
        ----------
        callbacks : callable or list of callables, optional
            Function(s) called after each evaluation with the record dict.
        """
        if callbacks is None:
            self._callbacks = []
        elif callable(callbacks):
            self._callbacks = [callbacks]
        else:
            self._callbacks = list(callbacks)

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback to be invoked after each evaluation.

        Parameters
        ----------
        callback : callable
            Function that takes a record dict with parameters and 'score'.

        Examples
        --------
        >>> func = SphereFunction(n_dim=2)
        >>> func.add_callback(lambda r: print(f"Score: {r['score']}"))
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a previously added callback.

        Parameters
        ----------
        callback : callable
            The callback to remove.

        Raises
        ------
        ValueError
            If the callback is not in the list.
        """
        self._callbacks.remove(callback)

    def clear_callbacks(self) -> None:
        """Remove all callbacks."""
        self._callbacks = []

    @property
    def callbacks(self) -> List[Callable[[Dict[str, Any]], None]]:
        """List of registered callbacks (read-only copy)."""
        return self._callbacks.copy()
