# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for catch_errors functionality in test functions."""

import math

import pytest

from surfaces.test_functions.algebraic import SphereFunction
from surfaces.test_functions.algebraic.constrained import CantileverBeamFunction


class FailingSphereFunction(SphereFunction):
    """Test function that raises different errors based on input."""

    def _objective(self, params):
        x = params["x0"]
        if x < -10:
            raise RuntimeError("x0 too negative")
        if x < 0:
            raise ValueError("x0 must be non-negative")
        if x == 0:
            return 1 / x  # ZeroDivisionError
        return sum(params[f"x{i}"] ** 2 for i in range(self.n_dim))


class TestCatchErrorsBasic:
    """Test basic catch_errors functionality."""

    def test_catch_errors_default_none(self):
        """By default, catch_errors is None (disabled)."""
        func = FailingSphereFunction(n_dim=2)
        assert func._error_handlers is None

    def test_error_propagates_when_disabled(self):
        """Errors propagate when catch_errors is None."""
        func = FailingSphereFunction(n_dim=2, catch_errors=None)

        with pytest.raises(ValueError, match="x0 must be non-negative"):
            func({"x0": -1.0, "x1": 0.0})

    def test_catch_specific_error_type(self):
        """Specific error type returns mapped value."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={ValueError: 999.0},
        )

        result = func({"x0": -1.0, "x1": 0.0})
        assert result == 999.0

    def test_uncaught_error_propagates(self):
        """Errors not in dict still propagate."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={ValueError: 999.0},  # Only catches ValueError
        )

        # RuntimeError not in dict, should propagate
        with pytest.raises(RuntimeError, match="x0 too negative"):
            func({"x0": -15.0, "x1": 0.0})

    def test_multiple_error_types(self):
        """Multiple error types with different return values."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={
                ValueError: 100.0,
                RuntimeError: 200.0,
                ZeroDivisionError: 300.0,
            },
        )

        assert func({"x0": -1.0, "x1": 0.0}) == 100.0  # ValueError
        assert func({"x0": -15.0, "x1": 0.0}) == 200.0  # RuntimeError
        assert func({"x0": 0.0, "x1": 0.0}) == 300.0  # ZeroDivisionError

    def test_catch_base_exception_class(self):
        """Using Exception catches all exception types."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={Exception: float("inf")},
        )

        # All errors return inf
        assert func({"x0": -1.0, "x1": 0.0}) == float("inf")  # ValueError
        assert func({"x0": -15.0, "x1": 0.0}) == float("inf")  # RuntimeError
        assert func({"x0": 0.0, "x1": 0.0}) == float("inf")  # ZeroDivisionError

    def test_normal_evaluation_unaffected(self):
        """Normal evaluations work correctly with catch_errors set."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={ValueError: float("inf")},
        )

        result = func({"x0": 1.0, "x1": 2.0})
        assert result == 5.0  # 1^2 + 2^2


class TestCatchErrorsReturnValues:
    """Test various return values for caught errors."""

    def test_return_inf(self):
        """Can return infinity."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={ValueError: float("inf")},
        )
        result = func({"x0": -1.0, "x1": 0.0})
        assert result == float("inf")
        assert math.isinf(result)

    def test_return_negative_inf(self):
        """Can return negative infinity."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={ValueError: float("-inf")},
        )
        result = func({"x0": -1.0, "x1": 0.0})
        assert result == float("-inf")

    def test_return_nan(self):
        """Can return NaN."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={ValueError: float("nan")},
        )
        result = func({"x0": -1.0, "x1": 0.0})
        assert math.isnan(result)

    def test_return_custom_value(self):
        """Can return any custom numeric value."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={ValueError: 42.5},
        )
        result = func({"x0": -1.0, "x1": 0.0})
        assert result == 42.5


class TestCatchErrorsWithDataCollection:
    """Test catch_errors interaction with data collection."""

    def test_error_result_recorded_in_search_data(self):
        """Error return values are recorded in search_data."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={ValueError: 999.0},
            collect_data=True,
        )

        func({"x0": -1.0, "x1": 0.0})

        assert len(func.data.search_data) == 1
        assert func.data.search_data[0]["score"] == 999.0

    def test_error_does_not_become_best_for_minimize(self):
        """High error return values don't become best_score for minimize."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={ValueError: float("inf")},
        )

        func({"x0": 1.0, "x1": 1.0})  # Normal: score = 2.0
        func({"x0": -1.0, "x1": 0.0})  # Error: score = inf

        assert func.data.best_score == 2.0
        assert func.data.n_evaluations == 2


class TestCatchErrorsWithMemory:
    """Test catch_errors interaction with memory caching."""

    def test_error_result_cached(self):
        """Error return values are cached in memory."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={ValueError: 999.0},
            memory=True,
        )

        result1 = func({"x0": -1.0, "x1": 0.0})
        result2 = func({"x0": -1.0, "x1": 0.0})  # Same position

        assert result1 == 999.0
        assert result2 == 999.0
        assert len(func._memory_cache) == 1


class TestCatchErrorsWithCallbacks:
    """Test catch_errors interaction with callbacks."""

    def test_callback_receives_error_return_value(self):
        """Callbacks receive the error return value as score."""
        records = []
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={ValueError: 999.0},
            callbacks=lambda r: records.append(r),
        )

        func({"x0": -1.0, "x1": 0.0})

        assert len(records) == 1
        assert records[0]["score"] == 999.0


class TestCatchErrorsInheritance:
    """Test exception inheritance behavior."""

    def test_catches_subclass_with_base_class(self):
        """Exception subclasses are caught by base class entry."""

        class CustomValueError(ValueError):
            pass

        class SubclassErrorFunc(SphereFunction):
            def _objective(self, params):
                raise CustomValueError("custom error")

        func = SubclassErrorFunc(
            n_dim=2,
            catch_errors={ValueError: 123.0},  # Base class
        )

        # CustomValueError is subclass of ValueError, should be caught
        result = func({"x0": 1.0, "x1": 1.0})
        assert result == 123.0

    def test_more_specific_match_first(self):
        """When multiple matches possible, first match in dict wins."""

        class SpecificError(ValueError):
            pass

        class MultiMatchFunc(SphereFunction):
            def _objective(self, params):
                raise SpecificError("specific")

        # Note: dict iteration order is preserved in Python 3.7+
        func = MultiMatchFunc(
            n_dim=2,
            catch_errors={
                SpecificError: 100.0,  # More specific first
                ValueError: 200.0,  # Base class second
            },
        )

        result = func({"x0": 1.0, "x1": 1.0})
        assert result == 100.0  # First match wins


class TestCatchErrorsEllipsisCatchAll:
    """Test Ellipsis (...) as catch-all key in catch_errors."""

    def test_ellipsis_catches_all_errors(self):
        """Ellipsis catches any exception type."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={...: float("inf")},
        )

        # All error types return inf
        assert func({"x0": -1.0, "x1": 0.0}) == float("inf")  # ValueError
        assert func({"x0": -15.0, "x1": 0.0}) == float("inf")  # RuntimeError
        assert func({"x0": 0.0, "x1": 0.0}) == float("inf")  # ZeroDivisionError

    def test_ellipsis_with_specific_exceptions(self):
        """Specific exceptions take precedence when listed before Ellipsis."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={
                ValueError: 100.0,  # Specific handling
                ...: float("inf"),  # Everything else
            },
        )

        assert func({"x0": -1.0, "x1": 0.0}) == 100.0  # ValueError -> specific
        assert func({"x0": -15.0, "x1": 0.0}) == float("inf")  # RuntimeError -> catch-all
        assert func({"x0": 0.0, "x1": 0.0}) == float("inf")  # ZeroDivisionError -> catch-all

    def test_ellipsis_order_matters(self):
        """First match wins - Ellipsis first catches everything."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={
                ...: 999.0,  # Catch-all first
                ValueError: 100.0,  # Never reached
            },
        )

        # Ellipsis matches first, so ValueError also returns 999.0
        assert func({"x0": -1.0, "x1": 0.0}) == 999.0

    def test_ellipsis_is_valid_dict_key(self):
        """Ellipsis can be used as a dictionary key."""
        catch_dict = {...: float("inf")}
        assert ... in catch_dict
        assert catch_dict[...] == float("inf")

    def test_ellipsis_with_nan_return(self):
        """Ellipsis catch-all can return NaN."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={...: float("nan")},
        )

        result = func({"x0": -1.0, "x1": 0.0})
        assert math.isnan(result)

    def test_normal_evaluation_with_ellipsis(self):
        """Normal evaluations work correctly with Ellipsis catch-all."""
        func = FailingSphereFunction(
            n_dim=2,
            catch_errors={...: float("inf")},
        )

        # Normal evaluation still works
        result = func({"x0": 1.0, "x1": 2.0})
        assert result == 5.0  # 1^2 + 2^2


class TestCatchErrorsWithEngineeringFunctions:
    """Test catch_errors with engineering functions."""

    def test_engineering_function_accepts_catch_errors(self):
        """Engineering functions accept catch_errors parameter."""
        func = CantileverBeamFunction(
            catch_errors={ValueError: float("inf")},
        )
        assert func._error_handlers == {ValueError: float("inf")}
