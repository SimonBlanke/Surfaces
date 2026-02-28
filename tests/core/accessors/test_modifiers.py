"""Tests for ModifierAccessor: Sequence-like access to modifiers."""

import collections.abc

import pytest

from surfaces.modifiers import GaussianNoise
from surfaces.test_functions._accessors._modifiers import ModifierAccessor
from surfaces.test_functions.algebraic import SphereFunction


class TestModifierInit:
    """Test ModifierAccessor initialization."""

    def test_no_modifiers(self):
        """No modifiers by default."""
        func = SphereFunction(n_dim=2)
        assert len(func.modifiers) == 0

    def test_init_with_modifier_list(self):
        """Modifiers passed at init are accessible."""
        noise = GaussianNoise(sigma=0.1, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[noise])
        assert len(func.modifiers) == 1


class TestModifierEffect:
    """Test that modifiers affect evaluation results."""

    def test_modifier_alters_result(self):
        """A modifier changes the function output."""
        noise = GaussianNoise(sigma=1.0, seed=42)
        func_pure = SphereFunction(n_dim=2)
        func_noisy = SphereFunction(n_dim=2, modifiers=[noise])

        result_pure = func_pure([0.0, 0.0])
        result_noisy = func_noisy([0.0, 0.0])

        # Sphere(0,0) = 0.0; with noise it should differ
        assert result_pure == 0.0
        assert result_noisy != 0.0

    def test_pure_bypasses_modifiers(self):
        """func.pure() returns the unmodified value."""
        noise = GaussianNoise(sigma=1.0, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[noise])

        result = func.pure([1.0, 2.0])
        assert result == 5.0  # 1^2 + 2^2, no noise

    def test_pure_does_not_update_search_data(self):
        """func.pure() does not record an evaluation."""
        func = SphereFunction(n_dim=2)

        func.pure([1.0, 2.0])

        assert func.data.n_evaluations == 0
        assert func.data.search_data == []


class TestModifierManagement:
    """Test add, remove, clear, and reset operations."""

    def test_add(self):
        """add() appends a modifier."""
        func = SphereFunction(n_dim=2)
        noise = GaussianNoise(sigma=0.1, seed=42)
        func.modifiers.add(noise)
        assert len(func.modifiers) == 1

    def test_remove(self):
        """remove() removes a specific modifier."""
        noise = GaussianNoise(sigma=0.1, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[noise])
        func.modifiers.remove(noise)
        assert len(func.modifiers) == 0

    def test_remove_unknown_raises_value_error(self):
        """remove() raises ValueError for unknown modifier."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(ValueError):
            func.modifiers.remove(GaussianNoise(sigma=0.1))

    def test_clear(self):
        """clear() removes all modifiers."""
        func = SphereFunction(
            n_dim=2,
            modifiers=[GaussianNoise(sigma=0.1), GaussianNoise(sigma=0.2)],
        )
        func.modifiers.clear()
        assert len(func.modifiers) == 0

    def test_reset_resets_modifier_state(self):
        """reset() calls reset() on each modifier."""
        noise = GaussianNoise(sigma=0.1, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[noise])

        # Trigger a few evaluations to advance the modifier's internal state
        func([1.0, 1.0])
        func([2.0, 2.0])
        assert noise.evaluation_count == 2

        func.modifiers.reset()
        assert noise.evaluation_count == 0


class TestModifierSequenceProtocol:
    """Test that ModifierAccessor implements collections.abc.Sequence."""

    def test_isinstance_sequence(self):
        """ModifierAccessor is a Sequence."""
        func = SphereFunction(n_dim=2)
        assert isinstance(func.modifiers, collections.abc.Sequence)

    def test_len(self):
        """len() returns the number of modifiers."""
        func = SphereFunction(
            n_dim=2,
            modifiers=[GaussianNoise(sigma=0.1), GaussianNoise(sigma=0.2)],
        )
        assert len(func.modifiers) == 2

    def test_getitem(self):
        """Indexing returns the modifier at that position."""
        noise = GaussianNoise(sigma=0.1, seed=42)
        func = SphereFunction(n_dim=2, modifiers=[noise])
        assert func.modifiers[0] is noise

    def test_iter(self):
        """Iteration yields all modifiers."""
        n1 = GaussianNoise(sigma=0.1)
        n2 = GaussianNoise(sigma=0.2)
        func = SphereFunction(n_dim=2, modifiers=[n1, n2])
        result = list(func.modifiers)
        assert result == [n1, n2]


class TestModifierCaching:
    """Test accessor caching on the function instance."""

    def test_accessor_is_cached(self):
        """Repeated access returns the same ModifierAccessor instance."""
        func = SphereFunction(n_dim=2)
        assert func.modifiers is func.modifiers

    def test_accessor_type(self):
        """func.modifiers is a ModifierAccessor."""
        func = SphereFunction(n_dim=2)
        assert isinstance(func.modifiers, ModifierAccessor)
