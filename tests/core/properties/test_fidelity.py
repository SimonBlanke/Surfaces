"""Tests for the fidelity parameter on BaseTestFunction.

These tests verify the fidelity parameter mechanics at the base class
level using algebraic functions (no sklearn required).
"""

import pytest

from surfaces.test_functions.algebraic import RastriginFunction, SphereFunction


class TestFidelityValidation:
    """Verify fidelity input validation."""

    def test_fidelity_none_is_default(self):
        func = SphereFunction(n_dim=2)
        result_no_fidelity = func({"x0": 1.0, "x1": 2.0})
        result_none = func({"x0": 1.0, "x1": 2.0}, fidelity=None)
        assert result_no_fidelity == result_none

    def test_fidelity_rejects_zero(self):
        func = SphereFunction(n_dim=2)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
            func({"x0": 1.0, "x1": 2.0}, fidelity=0.0)

    def test_fidelity_rejects_negative(self):
        func = SphereFunction(n_dim=2)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
            func({"x0": 1.0, "x1": 2.0}, fidelity=-0.5)

    def test_fidelity_rejects_above_one(self):
        func = SphereFunction(n_dim=2)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
            func({"x0": 1.0, "x1": 2.0}, fidelity=1.5)

    def test_fidelity_rejects_string(self):
        func = SphereFunction(n_dim=2)
        with pytest.raises(TypeError, match="must be a number"):
            func({"x0": 1.0, "x1": 2.0}, fidelity="high")

    def test_fidelity_accepts_one(self):
        func = SphereFunction(n_dim=2)
        result = func({"x0": 1.0, "x1": 2.0}, fidelity=1.0)
        assert isinstance(result, float)

    def test_fidelity_accepts_small_float(self):
        func = SphereFunction(n_dim=2)
        result = func({"x0": 1.0, "x1": 2.0}, fidelity=0.01)
        assert isinstance(result, float)

    def test_fidelity_is_keyword_only(self):
        """fidelity cannot be passed as positional argument."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(TypeError):
            func({"x0": 1.0, "x1": 2.0}, 0.5)


class TestFidelityAlgebraicPassthrough:
    """Verify that algebraic functions ignore fidelity (no effect on result)."""

    def test_fidelity_does_not_change_algebraic_result(self):
        func = SphereFunction(n_dim=2)
        params = {"x0": 1.0, "x1": 2.0}
        result_full = func(params)
        result_low = func(params, fidelity=0.1)
        assert result_full == result_low

    def test_fidelity_does_not_change_rastrigin_result(self):
        func = RastriginFunction(n_dim=3)
        params = {"x0": 0.5, "x1": -0.3, "x2": 1.2}
        result_full = func(params)
        result_low = func(params, fidelity=0.5)
        assert result_full == result_low


class TestFidelityCacheSeparation:
    """Verify that memory cache distinguishes fidelity levels."""

    def test_different_fidelity_different_cache_keys(self):
        func = SphereFunction(n_dim=2, memory=True)
        params = {"x0": 1.0, "x1": 2.0}

        func(params, fidelity=0.1)
        func(params, fidelity=0.5)
        func(params, fidelity=None)

        # For algebraic functions the results are identical, but the
        # cache should have 3 separate entries
        assert len(func._memory_cache) == 3

    def test_same_fidelity_hits_cache(self):
        func = SphereFunction(n_dim=2, memory=True)
        params = {"x0": 1.0, "x1": 2.0}

        func(params, fidelity=0.3)
        func(params, fidelity=0.3)

        assert len(func._memory_cache) == 1
        assert func.data.n_evaluations == 2


class TestFidelityDataCollection:
    """Verify that fidelity appears in search_data records."""

    def test_fidelity_in_record_when_set(self):
        func = SphereFunction(n_dim=2)
        func({"x0": 1.0, "x1": 2.0}, fidelity=0.3)
        record = func.data.search_data[-1]
        assert "fidelity" in record
        assert record["fidelity"] == 0.3

    def test_no_fidelity_key_when_none(self):
        func = SphereFunction(n_dim=2)
        func({"x0": 1.0, "x1": 2.0})
        record = func.data.search_data[-1]
        assert "fidelity" not in record

    def test_fidelity_in_callback_record(self):
        records = []
        func = SphereFunction(n_dim=2, callbacks=[records.append])
        func({"x0": 1.0, "x1": 2.0}, fidelity=0.7)
        assert records[-1]["fidelity"] == 0.7

    def test_no_fidelity_in_callback_when_none(self):
        records = []
        func = SphereFunction(n_dim=2, callbacks=[records.append])
        func({"x0": 1.0, "x1": 2.0})
        assert "fidelity" not in records[-1]


class TestFidelityPure:
    """Verify that pure() also accepts fidelity."""

    def test_pure_accepts_fidelity(self):
        func = SphereFunction(n_dim=2)
        result = func.pure({"x0": 1.0, "x1": 2.0}, fidelity=0.5)
        assert isinstance(result, float)

    def test_pure_fidelity_does_not_affect_algebraic(self):
        func = SphereFunction(n_dim=2)
        params = {"x0": 1.0, "x1": 2.0}
        assert func.pure(params) == func.pure(params, fidelity=0.3)
