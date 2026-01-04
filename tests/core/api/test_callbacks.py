# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for callback functionality in test functions."""

import pytest

from surfaces.test_functions import RastriginFunction, SphereFunction
from surfaces.test_functions.engineering import CantileverBeamFunction


class TestCallbackBasics:
    """Test basic callback functionality."""

    def test_single_callback(self):
        """Single callback is invoked with record dict."""
        records = []
        func = SphereFunction(n_dim=2, callbacks=lambda r: records.append(r))

        func([1.0, 2.0])

        assert len(records) == 1
        assert records[0]["x0"] == 1.0
        assert records[0]["x1"] == 2.0
        assert "score" in records[0]

    def test_callback_list(self):
        """Multiple callbacks are all invoked."""
        records1 = []
        records2 = []

        func = SphereFunction(
            n_dim=2,
            callbacks=[
                lambda r: records1.append(r),
                lambda r: records2.append(r),
            ],
        )

        func([1.0, 2.0])

        assert len(records1) == 1
        assert len(records2) == 1

    def test_callback_invoked_multiple_times(self):
        """Callback is invoked for each evaluation."""
        records = []
        func = SphereFunction(n_dim=2, callbacks=lambda r: records.append(r))

        func([1.0, 2.0])
        func([3.0, 4.0])
        func([5.0, 6.0])

        assert len(records) == 3
        assert records[0]["x0"] == 1.0
        assert records[1]["x0"] == 3.0
        assert records[2]["x0"] == 5.0

    def test_no_callback(self):
        """No callback is fine (default behavior)."""
        func = SphereFunction(n_dim=2)
        result = func([0.0, 0.0])
        assert result == 0.0


class TestCallbackManagement:
    """Test callback management methods."""

    def test_add_callback(self):
        """add_callback adds a callback after init."""
        func = SphereFunction(n_dim=2)
        records = []

        func.add_callback(lambda r: records.append(r))
        func([1.0, 2.0])

        assert len(records) == 1

    def test_remove_callback(self):
        """remove_callback removes a specific callback."""
        records = []
        callback = lambda r: records.append(r)

        func = SphereFunction(n_dim=2, callbacks=callback)
        func([1.0, 2.0])
        assert len(records) == 1

        func.remove_callback(callback)
        func([3.0, 4.0])
        assert len(records) == 1  # No new record

    def test_remove_callback_not_found(self):
        """remove_callback raises ValueError if callback not found."""
        func = SphereFunction(n_dim=2)

        with pytest.raises(ValueError):
            func.remove_callback(lambda r: None)

    def test_clear_callbacks(self):
        """clear_callbacks removes all callbacks."""
        records = []
        func = SphereFunction(
            n_dim=2,
            callbacks=[
                lambda r: records.append(r),
                lambda r: records.append(r),
            ],
        )

        func([1.0, 2.0])
        assert len(records) == 2

        func.clear_callbacks()
        func([3.0, 4.0])
        assert len(records) == 2  # No new records

    def test_callbacks_property(self):
        """callbacks property returns a copy of callback list."""
        callback1 = lambda r: None
        callback2 = lambda r: None

        func = SphereFunction(n_dim=2, callbacks=[callback1, callback2])

        callbacks = func.callbacks
        assert len(callbacks) == 2
        assert callback1 in callbacks
        assert callback2 in callbacks

        # Should be a copy, not the internal list
        callbacks.append(lambda r: None)
        assert len(func.callbacks) == 2


class TestCallbackWithDataCollection:
    """Test callback interaction with data collection."""

    def test_callback_with_collect_data_true(self):
        """Callbacks work alongside data collection."""
        records = []
        func = SphereFunction(n_dim=2, collect_data=True, callbacks=lambda r: records.append(r))

        func([1.0, 2.0])

        assert len(records) == 1
        assert func.n_evaluations == 1
        assert len(func.search_data) == 1

    def test_callback_with_collect_data_false(self):
        """Callbacks work even when data collection is disabled."""
        records = []
        func = SphereFunction(n_dim=2, collect_data=False, callbacks=lambda r: records.append(r))

        func([1.0, 2.0])

        assert len(records) == 1
        assert func.n_evaluations == 0  # Data collection disabled
        assert len(func.search_data) == 0

    def test_callback_with_memory(self):
        """Callbacks are invoked for cached results too."""
        records = []
        func = SphereFunction(n_dim=2, memory=True, callbacks=lambda r: records.append(r))

        func([1.0, 2.0])
        func([1.0, 2.0])  # Same position - cached

        assert len(records) == 2  # Callback still invoked for cached result


class TestCallbackWithDifferentFunctions:
    """Test callbacks work across different function types."""

    def test_callback_with_nd_function(self):
        """Callbacks work with N-dimensional functions."""
        records = []
        func = RastriginFunction(n_dim=5, callbacks=lambda r: records.append(r))

        func([0.0] * 5)

        assert len(records) == 1
        assert "x0" in records[0]
        assert "x4" in records[0]

    def test_callback_with_engineering_function(self):
        """Callbacks work with engineering functions."""
        records = []
        func = CantileverBeamFunction(callbacks=lambda r: records.append(r))

        func({"x1": 6.0, "x2": 5.3, "x3": 4.5, "x4": 3.5, "x5": 2.2})

        assert len(records) == 1
        assert records[0]["x1"] == 6.0
        assert "score" in records[0]


class TestCallbackEdgeCases:
    """Test edge cases for callbacks."""

    def test_callback_receives_correct_score(self):
        """Callback record contains the actual score returned."""
        records = []
        func = SphereFunction(n_dim=2, callbacks=lambda r: records.append(r))

        result = func([1.0, 2.0])

        assert records[0]["score"] == result

    def test_callback_with_maximize(self):
        """Callbacks work correctly with maximize objective."""
        records = []
        func = SphereFunction(n_dim=2, objective="maximize", callbacks=lambda r: records.append(r))

        result = func([1.0, 2.0])

        # Score in record should be the actual returned value
        assert records[0]["score"] == result

    def test_callback_exception_propagates(self):
        """Exception in callback propagates to caller."""

        def bad_callback(r):
            raise RuntimeError("Callback error")

        func = SphereFunction(n_dim=2, callbacks=bad_callback)

        with pytest.raises(RuntimeError, match="Callback error"):
            func([1.0, 2.0])
