"""Tests for CallbackAccessor: Sequence-like access to callbacks."""

import collections.abc

import pytest

from surfaces.test_functions._accessors._callbacks import CallbackAccessor
from surfaces.test_functions.algebraic import SphereFunction


class TestCallbackInit:
    """Test CallbackAccessor initialization with different callback configs."""

    def test_no_callbacks(self):
        """No callbacks by default."""
        func = SphereFunction(n_dim=2)
        assert len(func.callbacks) == 0

    def test_single_callback(self):
        """A single callable is wrapped into the accessor."""
        cb = lambda r: None
        func = SphereFunction(n_dim=2, callbacks=cb)
        assert len(func.callbacks) == 1

    def test_list_of_callbacks(self):
        """A list of callables is accepted."""
        cb1 = lambda r: None
        cb2 = lambda r: None
        func = SphereFunction(n_dim=2, callbacks=[cb1, cb2])
        assert len(func.callbacks) == 2


class TestCallbackManagement:
    """Test add, remove, and clear operations."""

    def test_add(self):
        """add() appends a callback."""
        func = SphereFunction(n_dim=2)
        cb = lambda r: None
        func.callbacks.add(cb)
        assert len(func.callbacks) == 1

    def test_remove(self):
        """remove() removes a specific callback."""
        cb = lambda r: None
        func = SphereFunction(n_dim=2, callbacks=cb)
        func.callbacks.remove(cb)
        assert len(func.callbacks) == 0

    def test_remove_unknown_raises_value_error(self):
        """remove() raises ValueError for unknown callback."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(ValueError):
            func.callbacks.remove(lambda r: None)

    def test_clear(self):
        """clear() removes all callbacks."""
        func = SphereFunction(n_dim=2, callbacks=[lambda r: None, lambda r: None])
        func.callbacks.clear()
        assert len(func.callbacks) == 0


class TestCallbackSequenceProtocol:
    """Test that CallbackAccessor implements collections.abc.Sequence."""

    def test_isinstance_sequence(self):
        """CallbackAccessor is a Sequence."""
        func = SphereFunction(n_dim=2)
        assert isinstance(func.callbacks, collections.abc.Sequence)

    def test_len(self):
        """len() returns the number of callbacks."""
        func = SphereFunction(n_dim=2, callbacks=[lambda r: None, lambda r: None])
        assert len(func.callbacks) == 2

    def test_getitem(self):
        """Indexing returns the callback at that position."""
        cb = lambda r: None
        func = SphereFunction(n_dim=2, callbacks=cb)
        assert func.callbacks[0] is cb

    def test_iter(self):
        """Iteration yields all callbacks."""
        cb1 = lambda r: None
        cb2 = lambda r: None
        func = SphereFunction(n_dim=2, callbacks=[cb1, cb2])
        result = list(func.callbacks)
        assert result == [cb1, cb2]


class TestCallbackCaching:
    """Test accessor caching on the function instance."""

    def test_accessor_is_cached(self):
        """Repeated access returns the same CallbackAccessor instance."""
        func = SphereFunction(n_dim=2)
        assert func.callbacks is func.callbacks

    def test_accessor_type(self):
        """func.callbacks is a CallbackAccessor."""
        func = SphereFunction(n_dim=2)
        assert isinstance(func.callbacks, CallbackAccessor)
