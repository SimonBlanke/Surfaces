# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for 2D plots (surface, contour, heatmap) via PlotAccessor.

These tests verify the 2D plot functionality through the accessor pattern.
"""

import pytest

from surfaces._visualize._param_resolver import resolve_params
from surfaces.test_functions.algebraic.standard.test_functions_2d import AckleyFunction
from surfaces.test_functions.algebraic.standard.test_functions_nd import SphereFunction


class TestSurfacePlotParamResolution:
    """Test parameter resolution for surface plot (requires 2 dims)."""

    def test_2d_function_uses_both_dims(self):
        """2D function should use both dimensions for plotting."""
        func = SphereFunction(n_dim=2)
        resolved = resolve_params(func, params=None, required_plot_dims=2)

        assert len(resolved.plot_dims) == 2
        assert len(resolved.fixed_dims) == 0
        assert resolved.plot_dim_names == ["x0", "x1"]

    def test_nd_function_uses_first_two_by_default(self):
        """N-D function should use first 2 dims by default."""
        func = SphereFunction(n_dim=5)
        resolved = resolve_params(func, params=None, required_plot_dims=2)

        assert len(resolved.plot_dims) == 2
        assert len(resolved.fixed_dims) == 3
        assert resolved.plot_dim_names == ["x0", "x1"]

    def test_explicit_dim_selection(self):
        """Can explicitly select which dimensions to plot."""
        func = SphereFunction(n_dim=5)
        resolved = resolve_params(
            func,
            params={"x1": ..., "x3": ...},
            required_plot_dims=2,
        )

        assert resolved.plot_dim_names == ["x1", "x3"]
        assert "x0" in resolved.fixed_dim_names
        assert "x2" in resolved.fixed_dim_names
        assert "x4" in resolved.fixed_dim_names

    def test_custom_bounds_for_dims(self):
        """Custom bounds should be used for dimensions."""
        func = SphereFunction(n_dim=2)
        resolved = resolve_params(
            func,
            params={"x0": (-2, 2), "x1": (-1, 1)},
            required_plot_dims=2,
        )

        x0_dim = resolved.get_dim("x0")
        x1_dim = resolved.get_dim("x1")
        assert x0_dim.bounds == (-2, 2)
        assert x1_dim.bounds == (-1, 1)

    def test_fixed_values_used_for_evaluation(self):
        """Fixed dimensions should have their values stored."""
        func = SphereFunction(n_dim=4)
        resolved = resolve_params(
            func,
            params={"x0": ..., "x1": ..., "x2": 1.5, "x3": -0.5},
            required_plot_dims=2,
        )

        assert resolved.fixed_values["x2"] == 1.5
        assert resolved.fixed_values["x3"] == -0.5


class TestSurfacePlotVisualization:
    """Test surface plot creation via accessor."""

    @pytest.fixture
    def plotly_available(self):
        """Check if plotly is available."""
        try:
            import plotly.graph_objects as go  # noqa: F401

            return True
        except ImportError:
            pytest.skip("plotly not installed")

    def test_surface_2d_function(self, plotly_available):
        """surface() works with 2D function."""
        func = AckleyFunction()
        fig = func.plot.surface()

        assert fig is not None
        # Surface plot has one Surface trace
        assert len(fig.data) == 1
        assert fig.data[0].type == "surface"

    def test_surface_nd_function_default(self, plotly_available):
        """surface() works with N-D function using default dims."""
        func = SphereFunction(n_dim=5)
        fig = func.plot.surface()

        assert fig is not None
        assert len(fig.data) == 1
        # Title should include "fixed" since dims are fixed
        assert "fixed" in fig.layout.title.text.lower()

    def test_surface_nd_function_explicit_dims(self, plotly_available):
        """surface() works with explicit dimension selection."""
        func = SphereFunction(n_dim=5)
        fig = func.plot.surface(params={"x1": ..., "x3": ...})

        assert fig is not None
        # Axes should be x1 and x3
        assert fig.layout.scene.xaxis.title.text == "x1"
        assert fig.layout.scene.yaxis.title.text == "x3"

    def test_surface_custom_resolution(self, plotly_available):
        """surface() respects resolution parameter."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.surface(resolution=25)

        assert fig is not None
        # Surface x and y should have 25 points
        assert len(fig.data[0].x) == 25
        assert len(fig.data[0].y) == 25

    def test_surface_wrong_dim_count_raises_error(self, plotly_available):
        """surface() raises error if not exactly 2 dims plotted."""
        func = SphereFunction(n_dim=5)
        with pytest.raises(ValueError, match="exactly 2"):
            func.plot.surface(params={"x0": ...})  # Only 1 dim


class TestContourPlotVisualization:
    """Test contour plot creation via accessor."""

    @pytest.fixture
    def plotly_available(self):
        """Check if plotly is available."""
        try:
            import plotly.graph_objects as go  # noqa: F401

            return True
        except ImportError:
            pytest.skip("plotly not installed")

    def test_contour_2d_function(self, plotly_available):
        """contour() works with 2D function."""
        func = AckleyFunction()
        fig = func.plot.contour()

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].type == "contour"

    def test_contour_nd_function(self, plotly_available):
        """contour() works with N-D function."""
        func = SphereFunction(n_dim=4)
        fig = func.plot.contour(params={"x0": ..., "x2": ...})

        assert fig is not None
        assert fig.layout.xaxis.title.text == "x0"
        assert fig.layout.yaxis.title.text == "x2"

    def test_contour_custom_resolution(self, plotly_available):
        """contour() respects resolution parameter."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.contour(resolution=30)

        assert fig is not None
        assert len(fig.data[0].x) == 30


class TestHeatmapPlotVisualization:
    """Test heatmap plot creation via accessor."""

    @pytest.fixture
    def plotly_available(self):
        """Check if plotly is available."""
        try:
            import plotly.graph_objects as go  # noqa: F401

            return True
        except ImportError:
            pytest.skip("plotly not installed")

    def test_heatmap_2d_function(self, plotly_available):
        """heatmap() works with 2D function."""
        func = AckleyFunction()
        fig = func.plot.heatmap()

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"

    def test_heatmap_nd_function(self, plotly_available):
        """heatmap() works with N-D function."""
        func = SphereFunction(n_dim=3)
        fig = func.plot.heatmap()

        assert fig is not None
        # Should fix x2 and plot x0, x1
        assert "fixed" in fig.layout.title.text.lower()

    def test_heatmap_custom_resolution(self, plotly_available):
        """heatmap() respects resolution parameter."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.heatmap(resolution=20)

        assert fig is not None
        assert len(fig.data[0].x) == 20


class TestMLFunctionWith2DPlots:
    """Test 2D plots with ML functions.

    These tests verify that the accessor pattern correctly handles
    ML function dimension names. They use param resolution only,
    since full evaluation requires surrogates or valid sklearn params.
    """

    @pytest.fixture
    def ml_func(self):
        """Create an ML function for testing."""
        try:
            from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.regression import (
                RandomForestRegressorFunction,
            )

            return RandomForestRegressorFunction(use_surrogate=True)
        except ImportError:
            pytest.skip("ML dependencies not installed")

    def test_ml_function_param_resolution(self, ml_func):
        """Param resolution works with ML dimension names."""
        resolved = resolve_params(ml_func, params=None, required_plot_dims=2)

        assert resolved.plot_dim_names == ["n_estimators", "max_depth"]
        assert "min_samples_split" in resolved.fixed_dim_names

    def test_ml_function_explicit_dim_resolution(self, ml_func):
        """Explicit dimension selection works with ML function."""
        resolved = resolve_params(
            ml_func,
            params={"n_estimators": ..., "min_samples_split": ...},
            required_plot_dims=2,
        )

        assert resolved.plot_dim_names == ["n_estimators", "min_samples_split"]
        assert "max_depth" in resolved.fixed_dim_names

    def test_ml_function_bounds_in_resolution(self, ml_func):
        """ML function bounds are correctly extracted."""
        resolved = resolve_params(ml_func, params=None, required_plot_dims=2)

        n_estimators_dim = resolved.get_dim("n_estimators")
        # n_estimators range is 10 to 190 in steps of 10
        assert n_estimators_dim.bounds[0] == 10
        assert n_estimators_dim.bounds[1] == 190
