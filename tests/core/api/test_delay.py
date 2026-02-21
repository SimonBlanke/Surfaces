# Tests for the DelayModifier


import time

import pytest

from surfaces.modifiers import DelayModifier
from surfaces.test_functions.algebraic import SphereFunction


class TestDelayModifier:
    """Tests for DelayModifier."""

    def test_basic_delay_application(self):
        """Test that delay is applied to evaluations."""
        delay = DelayModifier(delay=0.05)
        value = 5.0

        start = time.perf_counter()
        result = delay.apply(value, {}, {})
        elapsed = time.perf_counter() - start

        assert result == value  # Value should be unchanged
        assert elapsed >= 0.05  # Should have delayed

    def test_value_unchanged(self):
        """Test that the function value is not modified."""
        delay = DelayModifier(delay=0.01)

        for value in [0.0, 1.0, -5.5, 100.0, float("inf")]:
            result = delay.apply(value, {}, {})
            assert result == value

    def test_zero_delay(self):
        """Test that zero delay works correctly."""
        delay = DelayModifier(delay=0.0)
        value = 5.0

        start = time.perf_counter()
        result = delay.apply(value, {}, {})
        elapsed = time.perf_counter() - start

        assert result == value
        assert elapsed < 0.01  # Should be nearly instant

    def test_negative_delay_raises(self):
        """Test that negative delay raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            DelayModifier(delay=-0.1)

    def test_repr(self):
        """Test string representation."""
        delay = DelayModifier(delay=0.5)
        assert repr(delay) == "DelayModifier(delay=0.5)"


class TestDelayIntegration:
    """Integration tests for DelayModifier with test functions."""

    def test_delay_with_sphere_function(self):
        """Test DelayModifier integrated with SphereFunction."""
        func = SphereFunction(n_dim=2, modifiers=[DelayModifier(delay=0.05)])

        start = time.perf_counter()
        result = func({"x0": 1.0, "x1": 1.0})
        elapsed = time.perf_counter() - start

        assert result == 2.0  # Sphere function: 1^2 + 1^2 = 2
        assert elapsed >= 0.05

    def test_multiple_evaluations_delay(self):
        """Test that each evaluation is delayed."""
        func = SphereFunction(n_dim=2, modifiers=[DelayModifier(delay=0.02)])

        n_evals = 5
        start = time.perf_counter()
        for _ in range(n_evals):
            func({"x0": 0.0, "x1": 0.0})
        elapsed = time.perf_counter() - start

        # Total time should be at least n_evals * delay
        assert elapsed >= n_evals * 0.02

    def test_true_value_bypasses_delay(self):
        """Test that true_value() bypasses the delay modifier."""
        func = SphereFunction(n_dim=2, modifiers=[DelayModifier(delay=0.1)])

        start = time.perf_counter()
        result = func.pure({"x0": 1.0, "x1": 1.0})
        elapsed = time.perf_counter() - start

        assert result == 2.0
        assert elapsed < 0.05  # Should be fast (no delay)
