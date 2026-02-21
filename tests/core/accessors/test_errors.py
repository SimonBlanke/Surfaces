"""Tests for ErrorAccessor: mapping-like access to error handlers."""

import pytest

from surfaces.test_functions._accessors._errors import ErrorAccessor
from surfaces.test_functions.algebraic import SphereFunction


class TestErrorInit:
    """Test ErrorAccessor initialization."""

    def test_no_error_handlers(self):
        """No handlers by default."""
        func = SphereFunction(n_dim=2)
        assert len(func.errors) == 0

    def test_init_with_handlers(self):
        """catch_errors dict is reflected in the accessor."""
        func = SphereFunction(n_dim=2, catch_errors={ValueError: 0.0})
        assert len(func.errors) == 1
        assert ValueError in func.errors

    def test_getitem(self):
        """Bracket access returns the return value for an exception type."""
        func = SphereFunction(n_dim=2, catch_errors={ValueError: -1.0})
        assert func.errors[ValueError] == -1.0


class TestErrorManagement:
    """Test add, remove, and clear operations."""

    def test_add(self):
        """add() registers a new error handler."""
        func = SphereFunction(n_dim=2)
        func.errors.add(TypeError, -1.0)
        assert len(func.errors) == 1
        assert TypeError in func.errors
        assert func.errors[TypeError] == -1.0

    def test_remove(self):
        """remove() unregisters an error handler."""
        func = SphereFunction(n_dim=2, catch_errors={ValueError: 0.0})
        func.errors.remove(ValueError)
        assert len(func.errors) == 0

    def test_remove_not_present_raises_key_error(self):
        """remove() raises KeyError for absent exception type."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(KeyError):
            func.errors.remove(KeyError)

    def test_clear(self):
        """clear() removes all handlers."""
        func = SphereFunction(
            n_dim=2,
            catch_errors={ValueError: 0.0, TypeError: -1.0},
        )
        func.errors.clear()
        assert len(func.errors) == 0


class TestErrorMappingProtocol:
    """Test mapping-like protocol on ErrorAccessor."""

    def test_len(self):
        """len() returns handler count."""
        func = SphereFunction(
            n_dim=2,
            catch_errors={ValueError: 0.0, TypeError: -1.0},
        )
        assert len(func.errors) == 2

    def test_contains(self):
        """'in' checks for exception type presence."""
        func = SphereFunction(n_dim=2, catch_errors={ValueError: 0.0})
        assert ValueError in func.errors
        assert TypeError not in func.errors

    def test_iter(self):
        """Iteration yields the registered exception types."""
        func = SphereFunction(
            n_dim=2,
            catch_errors={ValueError: 0.0, TypeError: -1.0},
        )
        types = list(func.errors)
        assert set(types) == {ValueError, TypeError}

    def test_getitem_missing_raises_key_error(self):
        """Bracket access raises KeyError for absent type."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(KeyError):
            func.errors[RuntimeError]


class TestErrorHandlingBehavior:
    """Test that configured error handlers actually intercept exceptions."""

    def test_caught_error_returns_configured_value(self):
        """When pure_objective_function raises, the handler return value is used."""
        func = SphereFunction(n_dim=2, catch_errors={ValueError: 999.0})

        # Patch the objective to raise ValueError
        def raising_objective(params):
            raise ValueError("bad input")

        func.pure_objective_function = raising_objective

        result = func([1.0, 2.0])
        assert result == 999.0

    def test_uncaught_error_propagates(self):
        """Errors not in the handler dict propagate normally."""
        func = SphereFunction(n_dim=2, catch_errors={ValueError: 0.0})

        def raising_objective(params):
            raise TypeError("wrong type")

        func.pure_objective_function = raising_objective

        with pytest.raises(TypeError, match="wrong type"):
            func([1.0, 2.0])

    def test_ellipsis_catches_all(self):
        """Ellipsis (...) as key is a catch-all for any exception."""
        func = SphereFunction(n_dim=2, catch_errors={...: -999.0})

        def raising_objective(params):
            raise RuntimeError("unexpected")

        func.pure_objective_function = raising_objective

        result = func([1.0, 2.0])
        assert result == -999.0


class TestErrorCaching:
    """Test accessor caching on the function instance."""

    def test_accessor_is_cached(self):
        """Repeated access returns the same ErrorAccessor instance."""
        func = SphereFunction(n_dim=2)
        assert func.errors is func.errors

    def test_accessor_type(self):
        """func.errors is an ErrorAccessor."""
        func = SphereFunction(n_dim=2)
        assert isinstance(func.errors, ErrorAccessor)
