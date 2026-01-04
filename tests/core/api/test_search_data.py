"""
Tests for the search data collection feature (collect_data parameter).

These tests verify the tracking of evaluation history, best results,
and timing information.
"""

import numpy as np

from surfaces.test_functions import SphereFunction


class TestSearchDataCollection:
    """Test that evaluation data is collected correctly."""

    def test_n_evaluations_tracked(self):
        """Test that n_evaluations increments correctly."""
        func = SphereFunction(n_dim=2)
        assert func.n_evaluations == 0

        func([0.0, 0.0])
        assert func.n_evaluations == 1

        func([1.0, 1.0])
        assert func.n_evaluations == 2

        func([2.0, 2.0])
        assert func.n_evaluations == 3

    def test_search_data_recorded(self):
        """Test that search_data contains evaluation records."""
        func = SphereFunction(n_dim=2)

        func({"x0": 1.0, "x1": 2.0})
        func({"x0": 0.0, "x1": 0.0})

        assert len(func.search_data) == 2

        # Check first record
        assert func.search_data[0]["x0"] == 1.0
        assert func.search_data[0]["x1"] == 2.0
        assert "score" in func.search_data[0]

        # Check second record
        assert func.search_data[1]["x0"] == 0.0
        assert func.search_data[1]["x1"] == 0.0

    def test_best_score_tracked_minimize(self):
        """Test that best_score is tracked for minimization."""
        func = SphereFunction(n_dim=2, objective="minimize")

        func([5.0, 5.0])  # score = 50
        assert func.best_score == 50.0

        func([1.0, 1.0])  # score = 2 (better)
        assert func.best_score == 2.0

        func([3.0, 3.0])  # score = 18 (worse, should not update)
        assert func.best_score == 2.0

    def test_best_score_tracked_maximize(self):
        """Test that best_score is tracked for maximization."""
        func = SphereFunction(n_dim=2, objective="maximize")

        func([1.0, 1.0])  # score = -2
        assert func.best_score == -2.0

        func([5.0, 5.0])  # score = -50 (worse for maximize)
        assert func.best_score == -2.0

        func([0.5, 0.5])  # score = -0.5 (better for maximize)
        assert func.best_score == -0.5

    def test_best_params_tracked(self):
        """Test that best_params is updated with best result."""
        func = SphereFunction(n_dim=2)

        func([5.0, 5.0])
        assert func.best_params == {"x0": 5.0, "x1": 5.0}

        func([1.0, 2.0])  # better
        assert func.best_params == {"x0": 1.0, "x1": 2.0}

        func([0.0, 0.0])  # best
        assert func.best_params == {"x0": 0.0, "x1": 0.0}

    def test_total_time_tracked(self):
        """Test that total_time accumulates."""
        func = SphereFunction(n_dim=2, sleep=0.01)

        func([0.0, 0.0])
        assert func.total_time >= 0.01

        func([1.0, 1.0])
        assert func.total_time >= 0.02


class TestCollectDataDisabled:
    """Test behavior when collect_data=False."""

    def test_no_data_collected(self):
        """Test that no data is collected when disabled."""
        func = SphereFunction(n_dim=2, collect_data=False)

        func([1.0, 1.0])
        func([2.0, 2.0])

        assert func.n_evaluations == 0
        assert func.search_data == []
        assert func.best_score is None
        assert func.best_params is None
        assert func.total_time == 0.0


class TestResetMethods:
    """Test the reset methods."""

    def test_reset_data(self):
        """Test reset_data clears evaluation data but not memory cache."""
        func = SphereFunction(n_dim=2, memory=True)

        func([1.0, 1.0])
        func([2.0, 2.0])

        assert func.n_evaluations == 2
        assert len(func._memory_cache) == 2

        func.reset_data()

        assert func.n_evaluations == 0
        assert func.search_data == []
        assert func.best_score is None
        assert func.best_params is None
        assert func.total_time == 0.0
        # Memory cache should still be intact
        assert len(func._memory_cache) == 2

    def test_reset_memory(self):
        """Test reset_memory clears only the memory cache."""
        func = SphereFunction(n_dim=2, memory=True)

        func([1.0, 1.0])
        func([2.0, 2.0])

        func.reset_memory()

        # Data should still be there
        assert func.n_evaluations == 2
        # Memory cache should be cleared
        assert len(func._memory_cache) == 0

    def test_reset_clears_everything(self):
        """Test reset clears both data and memory cache."""
        func = SphereFunction(n_dim=2, memory=True)

        func([1.0, 1.0])
        func([2.0, 2.0])

        func.reset()

        assert func.n_evaluations == 0
        assert func.search_data == []
        assert func.best_score is None
        assert func.best_params is None
        assert func.total_time == 0.0
        assert len(func._memory_cache) == 0


class TestMemoryAndSearchDataInteraction:
    """Test interaction between memory caching and search data."""

    def test_cached_evaluations_still_recorded(self):
        """Test that cached evaluations are still recorded in search_data."""
        func = SphereFunction(n_dim=2, memory=True)

        # First evaluation
        func([1.0, 1.0])
        assert func.n_evaluations == 1

        # Second evaluation (from cache)
        func([1.0, 1.0])
        assert func.n_evaluations == 2

        # Both should be in search_data
        assert len(func.search_data) == 2

    def test_cached_evaluations_no_time_added(self):
        """Test that cached evaluations don't add to total_time."""
        func = SphereFunction(n_dim=2, memory=True, sleep=0.05)

        func([1.0, 1.0])
        time_after_first = func.total_time

        # Cached call should not add significant time
        func([1.0, 1.0])
        time_after_second = func.total_time

        # Time should be nearly the same (cached call is instant)
        assert time_after_second - time_after_first < 0.01


class TestSearchDataFormat:
    """Test the format of search_data records."""

    def test_record_contains_all_params(self):
        """Test that records contain all parameter names."""
        func = SphereFunction(n_dim=3)

        func({"x0": 1.0, "x1": 2.0, "x2": 3.0})

        record = func.search_data[0]
        assert "x0" in record
        assert "x1" in record
        assert "x2" in record
        assert "score" in record

    def test_record_values_correct(self):
        """Test that record values are correct."""
        func = SphereFunction(n_dim=2)

        result = func([3.0, 4.0])  # 9 + 16 = 25

        record = func.search_data[0]
        assert record["x0"] == 3.0
        assert record["x1"] == 4.0
        assert record["score"] == result
        assert record["score"] == 25.0

    def test_array_input_recorded_as_dict(self):
        """Test that array inputs are recorded with proper param names."""
        func = SphereFunction(n_dim=2)

        func(np.array([1.0, 2.0]))

        record = func.search_data[0]
        assert record["x0"] == 1.0
        assert record["x1"] == 2.0


class TestCollectDataDefault:
    """Test that collect_data is True by default."""

    def test_collect_data_default_true(self):
        """Test that data collection is enabled by default."""
        func = SphereFunction(n_dim=2)
        assert func.collect_data is True

    def test_data_collected_by_default(self):
        """Test that data is collected without explicit parameter."""
        func = SphereFunction(n_dim=2)

        func([1.0, 1.0])

        assert func.n_evaluations == 1
        assert len(func.search_data) == 1
        assert func.best_score is not None
