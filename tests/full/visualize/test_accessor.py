# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for PlotAccessor class.

These tests verify the accessor infrastructure without requiring plotly.
They focus on:
- Accessor availability on test functions
- Method presence and signatures
- Default properties on base classes
"""

import pytest

from surfaces.test_functions.algebraic.standard.test_functions_2d import AckleyFunction
from surfaces.test_functions.algebraic.standard.test_functions_nd import SphereFunction
from surfaces.visualize._accessor import PlotAccessor


class TestPlotAccessorAvailability:
    """Test that PlotAccessor is available on test functions."""

    def test_plot_property_exists(self):
        """Test functions have a plot property."""
        func = SphereFunction(n_dim=2)
        assert hasattr(func, "plot")

    def test_plot_returns_accessor(self):
        """plot property returns a PlotAccessor instance."""
        func = SphereFunction(n_dim=2)
        assert isinstance(func.plot, PlotAccessor)

    def test_accessor_has_function_reference(self):
        """PlotAccessor stores reference to the test function."""
        func = SphereFunction(n_dim=2)
        accessor = func.plot
        assert accessor._func is func

    def test_dimensions_property_exists(self):
        """Test functions have a dimensions property."""
        func = SphereFunction(n_dim=3)
        assert hasattr(func, "dimensions")
        assert func.dimensions == ["x0", "x1", "x2"]


class TestPlotAccessorMethods:
    """Test that PlotAccessor has expected methods."""

    def test_has_surface_method(self):
        """Accessor has surface method."""
        func = SphereFunction(n_dim=2)
        assert hasattr(func.plot, "surface")
        assert callable(func.plot.surface)

    def test_has_contour_method(self):
        """Accessor has contour method."""
        func = SphereFunction(n_dim=2)
        assert hasattr(func.plot, "contour")
        assert callable(func.plot.contour)

    def test_has_heatmap_method(self):
        """Accessor has heatmap method."""
        func = SphereFunction(n_dim=2)
        assert hasattr(func.plot, "heatmap")
        assert callable(func.plot.heatmap)

    def test_has_multi_slice_method(self):
        """Accessor has multi_slice method."""
        func = SphereFunction(n_dim=2)
        assert hasattr(func.plot, "multi_slice")
        assert callable(func.plot.multi_slice)

    def test_has_fitness_distribution_method(self):
        """Accessor has fitness_distribution method."""
        func = SphereFunction(n_dim=2)
        assert hasattr(func.plot, "fitness_distribution")
        assert callable(func.plot.fitness_distribution)

    def test_has_convergence_method(self):
        """Accessor has convergence method."""
        func = SphereFunction(n_dim=2)
        assert hasattr(func.plot, "convergence")
        assert callable(func.plot.convergence)

    def test_has_latex_method(self):
        """Accessor has latex method."""
        func = SphereFunction(n_dim=2)
        assert hasattr(func.plot, "latex")
        assert callable(func.plot.latex)

    def test_has_available_method(self):
        """Accessor has available method."""
        func = SphereFunction(n_dim=2)
        assert hasattr(func.plot, "available")
        assert callable(func.plot.available)

    def test_has_with_history_method(self):
        """Accessor has with_history method."""
        func = SphereFunction(n_dim=2)
        assert hasattr(func.plot, "with_history")
        assert callable(func.plot.with_history)


class TestWithHistoryChaining:
    """Test with_history method chaining."""

    def test_with_history_returns_accessor(self):
        """with_history returns the same accessor for chaining."""
        func = SphereFunction(n_dim=2)
        accessor = func.plot
        result = accessor.with_history([{"x0": 0, "x1": 0, "score": 0}])
        assert result is accessor

    def test_with_history_stores_history(self):
        """with_history stores the history data."""
        func = SphereFunction(n_dim=2)
        history = [{"x0": 0, "x1": 0, "score": 0}]
        accessor = func.plot.with_history(history)
        assert accessor._history is history


class TestAlgebraicFunctionDefaults:
    """Test default properties on AlgebraicFunction."""

    def test_default_plot_dims_2d(self):
        """2D function has correct default_plot_dims."""
        func = AckleyFunction()  # 2D
        assert hasattr(func, "default_plot_dims")
        assert func.default_plot_dims == ["x0", "x1"]

    def test_default_plot_dims_nd(self):
        """N-D function has first 2 dims as default."""
        func = SphereFunction(n_dim=5)
        assert func.default_plot_dims == ["x0", "x1"]

    def test_default_bounds_per_dim(self):
        """AlgebraicFunction has default_bounds_per_dim inferred from search_space."""
        func = SphereFunction(n_dim=3)
        assert hasattr(func, "default_bounds_per_dim")
        bounds = func.default_bounds_per_dim
        assert "x0" in bounds
        assert "x1" in bounds
        assert "x2" in bounds
        # Bounds are inferred from actual search_space values
        # Min should be close to -5, max close to 5 (depends on arange step)
        assert bounds["x0"][0] == pytest.approx(-5.0, abs=0.1)
        assert bounds["x0"][1] == pytest.approx(5.0, abs=0.5)

    def test_default_fixed(self):
        """AlgebraicFunction has default_fixed (middle value from search_space)."""
        func = SphereFunction(n_dim=3)
        assert hasattr(func, "default_fixed")
        fixed = func.default_fixed
        assert "x0" in fixed
        # Middle value from the search_space array (close to center)
        assert fixed["x0"] == pytest.approx(0.0, abs=0.5)

    def test_default_step(self):
        """AlgebraicFunction has default_step inferred from dimension type."""
        func = SphereFunction(n_dim=3)
        assert hasattr(func, "default_step")
        step = func.default_step
        assert "x0" in step
        # For dense numpy arrays: (max - min) / 100
        # Actual value depends on search_space size
        assert step["x0"] is not None
        assert step["x0"] > 0


class TestMachineLearningFunctionDefaults:
    """Test default properties on MachineLearningFunction."""

    @pytest.fixture
    def ml_func(self):
        """Create an ML function for testing."""
        # Import here to avoid issues if ML deps not installed
        try:
            from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.regression import (
                RandomForestRegressorFunction,
            )

            return RandomForestRegressorFunction(use_surrogate=True)
        except ImportError:
            pytest.skip("ML dependencies not installed")

    def test_default_plot_dims(self, ml_func):
        """ML function has default_plot_dims."""
        assert hasattr(ml_func, "default_plot_dims")
        dims = ml_func.default_plot_dims
        assert len(dims) == 2
        assert dims[0] == "n_estimators"
        assert dims[1] == "max_depth"

    def test_default_bounds_per_dim(self, ml_func):
        """ML function has default_bounds_per_dim."""
        assert hasattr(ml_func, "default_bounds_per_dim")
        bounds = ml_func.default_bounds_per_dim
        assert "n_estimators" in bounds
        assert "max_depth" in bounds
        # n_estimators ranges from 10 to 190 in steps of 10
        assert bounds["n_estimators"][0] == 10
        assert bounds["n_estimators"][1] == 190

    def test_default_fixed(self, ml_func):
        """ML function has default_fixed (middle value)."""
        import numpy as np

        assert hasattr(ml_func, "default_fixed")
        fixed = ml_func.default_fixed
        assert "n_estimators" in fixed
        assert "max_depth" in fixed
        assert "min_samples_split" in fixed
        # Middle value from the list (can be numpy type)
        assert isinstance(fixed["n_estimators"], (int, float, np.integer, np.floating))

    def test_default_step(self, ml_func):
        """ML function has default_step."""
        assert hasattr(ml_func, "default_step")
        step = ml_func.default_step
        assert "n_estimators" in step
        # n_estimators has step 10
        assert step["n_estimators"] == 10
