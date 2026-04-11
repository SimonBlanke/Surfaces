# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for the new CustomTestFunction module."""

import numpy as np
import pytest

from surfaces.custom_test_function import CustomTestFunction


@pytest.fixture
def sphere_func():
    """Simple sphere function for testing."""

    def sphere(params):
        return sum(v**2 for v in params.values())

    return CustomTestFunction(
        objective_fn=sphere,
        search_space={"x": (-5, 5), "y": (-5, 5)},
    )


@pytest.fixture
def sphere_func_with_data(sphere_func):
    """Sphere function with evaluation data."""
    np.random.seed(42)
    for _ in range(50):
        x, y = np.random.uniform(-5, 5, 2)
        sphere_func({"x": x, "y": y})
    return sphere_func


class TestBasicFunctionality:
    """Test basic CustomTestFunction behavior."""

    def test_call_with_dict(self, sphere_func):
        """Test calling with dict input."""
        result = sphere_func({"x": 1.0, "y": 2.0})
        assert result == 5.0

    def test_call_with_kwargs(self, sphere_func):
        """Test calling with kwargs."""
        result = sphere_func(x=1.0, y=2.0)
        assert result == 5.0

    def test_call_with_array(self, sphere_func):
        """Test calling with array input."""
        result = sphere_func(np.array([1.0, 2.0]))
        assert result == 5.0

    def test_call_with_list(self, sphere_func):
        """Test calling with list input."""
        result = sphere_func([1.0, 2.0])
        assert result == 5.0

    def test_n_evaluations(self, sphere_func):
        """Test evaluation counting."""
        assert sphere_func.n_evaluations == 0
        sphere_func(x=1, y=1)
        assert sphere_func.n_evaluations == 1
        sphere_func(x=2, y=2)
        sphere_func(x=3, y=3)
        assert sphere_func.n_evaluations == 3

    def test_search_data(self, sphere_func):
        """Test search data collection."""
        sphere_func(x=1, y=2)
        sphere_func(x=3, y=4)

        assert len(sphere_func.search_data) == 2
        assert sphere_func.search_data[0] == {"x": 1, "y": 2, "score": 5}
        assert sphere_func.search_data[1] == {"x": 3, "y": 4, "score": 25}

    def test_best_score(self, sphere_func):
        """Test best score tracking."""
        sphere_func(x=3, y=4)  # score = 25
        assert sphere_func.best_score == 25

        sphere_func(x=1, y=1)  # score = 2
        assert sphere_func.best_score == 2

        sphere_func(x=5, y=5)  # score = 50
        assert sphere_func.best_score == 2  # Still 2

    def test_best_params(self, sphere_func):
        """Test best params tracking."""
        sphere_func(x=3, y=4)
        sphere_func(x=1, y=1)

        assert sphere_func.best_params == {"x": 1, "y": 1}


class TestSearchSpace:
    """Test search space handling."""

    def test_bounds_tuple(self):
        """Test bounds as (min, max) tuple."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"x": (-5, 5), "y": (-10, 10)},
            resolution=50,
        )

        assert len(func.search_space["x"]) == 50
        assert func.search_space["x"][0] == -5
        assert func.search_space["x"][-1] == pytest.approx(5, rel=0.1)

    def test_explicit_array(self):
        """Test explicit array search space."""
        x_vals = np.array([1, 2, 3, 4, 5])

        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"x": x_vals},
        )

        np.testing.assert_array_equal(func.search_space["x"], x_vals)

    def test_n_dim(self):
        """Test n_dim property."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"a": (0, 1), "b": (0, 1), "c": (0, 1)},
        )
        assert func.n_dim == 3

    def test_param_names(self):
        """Test param_names are sorted."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"z": (0, 1), "a": (0, 1), "m": (0, 1)},
        )
        assert func.param_names == ["a", "m", "z"]

    def test_bounds_property(self):
        """Test bounds property."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"x": (-5, 5), "y": (0, 10)},
        )
        bounds = func.bounds
        assert bounds["x"] == pytest.approx((-5, 5), rel=0.1)
        assert bounds["y"] == pytest.approx((0, 10), rel=0.1)


class TestMemoryCache:
    """Test memory caching functionality."""

    def test_memory_caching(self):
        """Test that memory caching avoids recomputation."""
        call_count = [0]

        def counting_func(params):
            call_count[0] += 1
            return params["x"] ** 2

        func = CustomTestFunction(
            objective_fn=counting_func,
            search_space={"x": (-5, 5)},
            memory=True,
        )

        func(x=2)
        assert call_count[0] == 1

        func(x=2)  # Should use cache
        assert call_count[0] == 1

        func(x=3)  # New value
        assert call_count[0] == 2

    def test_memory_disabled(self):
        """Test that memory can be disabled."""
        call_count = [0]

        def counting_func(params):
            call_count[0] += 1
            return params["x"] ** 2

        func = CustomTestFunction(
            objective_fn=counting_func,
            search_space={"x": (-5, 5)},
            memory=False,
        )

        func(x=2)
        func(x=2)
        assert call_count[0] == 2


class TestNamespaces:
    """Test that namespaces are properly lazy-loaded."""

    def test_analysis_namespace_type(self, sphere_func):
        """Test analysis namespace is correct type."""
        from surfaces.custom_test_function._namespaces import AnalysisNamespace

        assert isinstance(sphere_func.analysis, AnalysisNamespace)

    def test_plot_namespace_type(self, sphere_func):
        """Test plot namespace is correct type."""
        from surfaces.custom_test_function._namespaces import PlotNamespace

        assert isinstance(sphere_func.plot, PlotNamespace)

    def test_namespace_lazy_loading(self):
        """Test that namespaces are lazily loaded."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"x": (0, 1)},
        )

        # Namespaces should not be initialized yet
        assert func._analysis is None
        assert func._plot is None

        # Access triggers initialization
        _ = func.analysis
        assert func._analysis is not None
        assert func._plot is None  # Still not initialized

        _ = func.plot
        assert func._plot is not None


class TestAnalysisNamespace:
    """Test analysis namespace methods."""

    def test_summary(self, sphere_func_with_data):
        """Test summary generation."""
        summary = sphere_func_with_data.analysis.summary()

        assert "n_evaluations" in summary
        assert "best_score" in summary
        assert "mean_score" in summary
        assert summary["n_evaluations"] == 50

    def test_parameter_importance(self, sphere_func_with_data):
        """Test parameter importance calculation."""
        importance = sphere_func_with_data.analysis.parameter_importance()

        assert "x" in importance
        assert "y" in importance
        assert sum(importance.values()) == pytest.approx(1.0)

    def test_convergence(self, sphere_func_with_data):
        """Test convergence analysis."""
        conv = sphere_func_with_data.analysis.convergence()

        assert "is_converged" in conv
        assert "best_at_eval" in conv
        assert "running_best" in conv

    def test_suggest_refined_space(self, sphere_func_with_data):
        """Test refined space suggestion."""
        refined = sphere_func_with_data.analysis.suggest_refined_space()

        assert "x" in refined
        assert "y" in refined
        assert len(refined["x"]) == 2  # (min, max)

    def test_insufficient_data_error(self, sphere_func):
        """Test error when insufficient data."""
        with pytest.raises(ValueError, match="requires at least"):
            sphere_func.analysis.parameter_importance()


class TestExperimentMetadata:
    """Test experiment metadata handling."""

    def test_experiment_name(self):
        """Test experiment name is stored."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"x": (0, 1)},
            experiment="my-experiment",
        )
        assert func.experiment == "my-experiment"

    def test_tags(self):
        """Test tags are stored."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"x": (0, 1)},
            tags=["test", "production"],
        )
        assert func.tags == ["test", "production"]

    def test_metadata(self):
        """Test metadata is stored."""
        func = CustomTestFunction(
            objective_fn=lambda p: 0,
            search_space={"x": (0, 1)},
            metadata={"version": "1.0", "author": "test"},
        )
        assert func.metadata["version"] == "1.0"


class TestStateManagement:
    """Test state management methods."""

    def test_reset(self, sphere_func_with_data):
        """Test reset clears all state."""
        assert sphere_func_with_data.n_evaluations > 0

        sphere_func_with_data.reset()

        assert sphere_func_with_data.n_evaluations == 0
        assert sphere_func_with_data.search_data == []
        assert sphere_func_with_data.best_score is None
        assert sphere_func_with_data.best_params is None

    def test_reset_cache(self):
        """Test reset_cache only clears cache."""
        func = CustomTestFunction(
            objective_fn=lambda p: p["x"] ** 2,
            search_space={"x": (-5, 5)},
            memory=True,
        )

        func(x=1)
        func(x=2)
        assert func.n_evaluations == 2

        func.reset_cache()

        assert func.n_evaluations == 2  # Data preserved
        assert func._memory_cache == {}  # Cache cleared

    def test_get_data_as_arrays(self, sphere_func_with_data):
        """Test conversion to arrays."""
        X, y = sphere_func_with_data.get_data_as_arrays()

        assert X.shape == (50, 2)
        assert y.shape == (50,)


class TestRepr:
    """Test string representation."""

    def test_repr_contains_info(self, sphere_func):
        """Test repr contains key information."""
        sphere_func(x=1, y=2)

        repr_str = repr(sphere_func)

        assert "CustomTestFunction" in repr_str
        assert "n_dim=2" in repr_str
        assert "n_evaluations=1" in repr_str
