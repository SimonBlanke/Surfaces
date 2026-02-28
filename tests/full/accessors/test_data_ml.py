"""Tests for DataAccessor with ML functions (requires sklearn)."""

from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction


class TestDataWithMLFunctions:
    """Test DataAccessor behavior with ML functions."""

    def test_ml_data_collection(self):
        """Data collection works for ML functions."""
        func = KNeighborsClassifierFunction(dataset="iris", cv=3)
        params = {k: list(v)[0] for k, v in func.search_space.items()}
        func(params)

        assert func.data.n_evaluations == 1
        assert len(func.data.search_data) == 1
        assert func.data.best_score is not None

    def test_ml_data_reset(self):
        """Data reset works for ML functions."""
        func = KNeighborsClassifierFunction(dataset="iris", cv=3)
        params = {k: list(v)[0] for k, v in func.search_space.items()}
        func(params)

        func.data.reset()
        assert func.data.n_evaluations == 0
        assert func.data.search_data == []
