"""Tests for DataAccessor: proxy to evaluation data on BaseTestFunction."""

from surfaces.test_functions._accessors._data import DataAccessor
from surfaces.test_functions.algebraic import SphereFunction


class TestDataInitialState:
    """Test DataAccessor initial state before any evaluations."""

    def test_n_evaluations_zero(self):
        """n_evaluations starts at zero."""
        func = SphereFunction(n_dim=2)
        assert func.data.n_evaluations == 0

    def test_search_data_empty(self):
        """search_data starts as an empty list."""
        func = SphereFunction(n_dim=2)
        assert func.data.search_data == []

    def test_best_score_none(self):
        """best_score starts as None."""
        func = SphereFunction(n_dim=2)
        assert func.data.best_score is None

    def test_best_params_none(self):
        """best_params starts as None."""
        func = SphereFunction(n_dim=2)
        assert func.data.best_params is None

    def test_total_time_zero(self):
        """total_time starts at 0.0."""
        func = SphereFunction(n_dim=2)
        assert func.data.total_time == 0.0


class TestDataAfterEvaluation:
    """Test DataAccessor state after function evaluations."""

    def test_n_evaluations_increments(self):
        """n_evaluations increments with each call."""
        func = SphereFunction(n_dim=2)

        func([1.0, 2.0])
        assert func.data.n_evaluations == 1

        func([3.0, 4.0])
        assert func.data.n_evaluations == 2

    def test_search_data_records_params_and_score(self):
        """search_data entries contain param values and score."""
        func = SphereFunction(n_dim=2)

        func({"x0": 1.0, "x1": 2.0})

        assert len(func.data.search_data) == 1
        record = func.data.search_data[0]
        assert record["x0"] == 1.0
        assert record["x1"] == 2.0
        assert record["score"] == 5.0  # 1^2 + 2^2

    def test_total_time_positive_after_evaluation(self):
        """total_time is positive after at least one evaluation."""
        func = SphereFunction(n_dim=2)
        func([0.0, 0.0])
        assert func.data.total_time >= 0.0


class TestDataBestScoreTracking:
    """Test best_score tracking for minimize and maximize objectives."""

    def test_best_score_minimize(self):
        """best_score tracks the minimum value for minimize objective."""
        func = SphereFunction(n_dim=2, objective="minimize")

        func([3.0, 4.0])  # score = 25
        assert func.data.best_score == 25.0

        func([1.0, 1.0])  # score = 2 (better)
        assert func.data.best_score == 2.0

        func([5.0, 5.0])  # score = 50 (worse, no update)
        assert func.data.best_score == 2.0

    def test_best_score_maximize(self):
        """best_score tracks the maximum value for maximize objective."""
        func = SphereFunction(n_dim=2, objective="maximize")

        func([1.0, 1.0])  # score = -2
        assert func.data.best_score == -2.0

        func([0.5, 0.5])  # score = -0.5 (better for maximize)
        assert func.data.best_score == -0.5

        func([3.0, 3.0])  # score = -18 (worse for maximize, no update)
        assert func.data.best_score == -0.5

    def test_best_params_updated(self):
        """best_params reflects the params that produced the best score."""
        func = SphereFunction(n_dim=2)

        func([3.0, 4.0])
        assert func.data.best_params == {"x0": 3.0, "x1": 4.0}

        func([0.0, 0.0])  # better
        assert func.data.best_params == {"x0": 0.0, "x1": 0.0}


class TestDataReset:
    """Test DataAccessor.reset() behavior."""

    def test_reset_clears_all_fields(self):
        """reset() sets everything back to initial state."""
        func = SphereFunction(n_dim=2)

        func([1.0, 1.0])
        func([2.0, 2.0])

        func.data.reset()

        assert func.data.n_evaluations == 0
        assert func.data.search_data == []
        assert func.data.best_score is None
        assert func.data.best_params is None
        assert func.data.total_time == 0.0


class TestDataCollectDataDisabled:
    """Test behavior when collect_data=False."""

    def test_no_tracking_when_disabled(self):
        """No counters update when collect_data=False."""
        func = SphereFunction(n_dim=2, collect_data=False)

        func([1.0, 1.0])
        func([2.0, 2.0])

        assert func.data.n_evaluations == 0
        assert func.data.search_data == []
        assert func.data.best_score is None
        assert func.data.best_params is None
        assert func.data.total_time == 0.0

    def test_callbacks_still_work_when_collect_data_false(self):
        """Callbacks fire even when data collection is off."""
        records = []
        func = SphereFunction(
            n_dim=2,
            collect_data=False,
            callbacks=lambda r: records.append(r),
        )

        func([1.0, 2.0])

        assert len(records) == 1
        assert records[0]["score"] == 5.0


class TestDataCaching:
    """Test DataAccessor caching on the function instance."""

    def test_accessor_is_cached(self):
        """Repeated access returns the same DataAccessor instance."""
        func = SphereFunction(n_dim=2)
        assert func.data is func.data

    def test_accessor_type(self):
        """func.data is a DataAccessor."""
        func = SphereFunction(n_dim=2)
        assert isinstance(func.data, DataAccessor)


class TestDataRepr:
    """Test DataAccessor string representation."""

    def test_repr_initial(self):
        """repr includes n_evaluations=0 and best_score=None."""
        func = SphereFunction(n_dim=2)
        r = repr(func.data)
        assert "n_evaluations=0" in r
        assert "best_score=None" in r

    def test_repr_after_evaluation(self):
        """repr reflects updated state after evaluation."""
        func = SphereFunction(n_dim=2)
        func([1.0, 2.0])
        r = repr(func.data)
        assert "n_evaluations=1" in r
        assert "best_score=5.0" in r
