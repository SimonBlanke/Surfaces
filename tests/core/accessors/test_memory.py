"""Tests for MemoryAccessor: cache management for test functions."""

from surfaces.test_functions._accessors._memory import MemoryAccessor
from surfaces.test_functions.algebraic import SphereFunction


class TestMemoryDisabled:
    """Test MemoryAccessor when memory=False (default)."""

    def test_enabled_is_false(self):
        """Memory is disabled by default."""
        func = SphereFunction(n_dim=2)
        assert func.memory.enabled is False

    def test_size_is_zero(self):
        """Cache size is zero when memory is disabled."""
        func = SphereFunction(n_dim=2)
        assert func.memory.size == 0


class TestMemoryEnabled:
    """Test MemoryAccessor when memory=True."""

    def test_enabled_is_true(self):
        """Memory is enabled when constructed with memory=True."""
        func = SphereFunction(n_dim=2, memory=True)
        assert func.memory.enabled is True

    def test_size_after_evaluation(self):
        """Cache gains one entry after a unique evaluation."""
        func = SphereFunction(n_dim=2, memory=True)

        func([1.0, 2.0])
        assert func.memory.size == 1

    def test_repeated_evaluation_cache_hit(self):
        """Repeated evaluation at same point is a cache hit.

        n_evaluations increases for both calls (data is still recorded),
        but the memory cache has only one unique entry.
        """
        func = SphereFunction(n_dim=2, memory=True)

        func([1.0, 2.0])
        func([1.0, 2.0])

        assert func.data.n_evaluations == 2
        assert func.memory.size == 1

    def test_different_points_grow_cache(self):
        """Different evaluation points each produce a cache entry."""
        func = SphereFunction(n_dim=2, memory=True)

        func([1.0, 2.0])
        func([3.0, 4.0])

        assert func.memory.size == 2


class TestMemoryReset:
    """Test MemoryAccessor.reset() behavior."""

    def test_reset_clears_cache(self):
        """reset() clears the memory cache."""
        func = SphereFunction(n_dim=2, memory=True)

        func([1.0, 2.0])
        assert func.memory.size == 1

        func.memory.reset()
        assert func.memory.size == 0


class TestMemoryRuntimeToggle:
    """Test enabling/disabling memory at runtime."""

    def test_enable_at_runtime(self):
        """Setting enabled=True activates caching mid-run."""
        func = SphereFunction(n_dim=2)
        assert func.memory.enabled is False

        func.memory.enabled = True
        assert func.memory.enabled is True

        func([1.0, 2.0])
        assert func.memory.size == 1

    def test_disable_at_runtime(self):
        """Setting enabled=False deactivates caching mid-run."""
        func = SphereFunction(n_dim=2, memory=True)

        func([1.0, 2.0])
        assert func.memory.size == 1

        func.memory.enabled = False

        # New evaluations should not be cached
        func([3.0, 4.0])
        assert func.memory.size == 1  # No new cache entry


class TestMemoryCaching:
    """Test accessor caching on the function instance."""

    def test_accessor_is_cached(self):
        """Repeated access returns the same MemoryAccessor instance."""
        func = SphereFunction(n_dim=2)
        assert func.memory is func.memory

    def test_accessor_type(self):
        """func.memory is a MemoryAccessor."""
        func = SphereFunction(n_dim=2)
        assert isinstance(func.memory, MemoryAccessor)
