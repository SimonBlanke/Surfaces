# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for visualization plot function APIs (legacy/bounds-based paths)."""

import numpy as np
import pytest

from surfaces.test_functions.algebraic import AckleyFunction, SphereFunction

# =============================================================================
# Contour Plot Tests (Legacy Path)
# =============================================================================


@pytest.mark.viz
class TestContourLegacy:
    """Test plot_contour with direct function + bounds (legacy path)."""

    def test_contour_2d_function(self):
        """Contour plot with 2D function."""
        from surfaces._visualize import plot_contour

        func = AckleyFunction()
        fig = plot_contour(func, resolution=10)
        assert fig is not None
        assert fig.data  # has traces

    def test_contour_custom_bounds(self):
        """Contour plot with custom bounds."""
        from surfaces._visualize import plot_contour

        func = AckleyFunction()
        bounds = {"x0": (-2.0, 2.0), "x1": (-2.0, 2.0)}
        fig = plot_contour(func, resolution=10, bounds=bounds)
        assert fig is not None

    def test_heatmap_2d_function(self):
        """Heatmap plot with 2D function."""
        from surfaces._visualize import plot_heatmap

        func = AckleyFunction()
        fig = plot_heatmap(func, resolution=10)
        assert fig is not None
        assert fig.data

    def test_contour_custom_title(self):
        """Contour plot with custom title."""
        from surfaces._visualize import plot_contour

        func = AckleyFunction()
        fig = plot_contour(func, resolution=10, title="Custom Title")
        assert "Custom Title" in fig.layout.title.text


# =============================================================================
# Surface Plot Tests (Legacy Path)
# =============================================================================


@pytest.mark.viz
class TestSurfaceLegacy:
    """Test plot_surface with direct function + bounds (legacy path)."""

    def test_surface_2d_function(self):
        """Surface plot with 2D function."""
        from surfaces._visualize import plot_surface

        func = AckleyFunction()
        fig = plot_surface(func, resolution=10)
        assert fig is not None
        assert fig.data

    def test_surface_with_contours(self):
        """Surface plot with contour projection."""
        from surfaces._visualize import plot_surface

        func = AckleyFunction()
        fig = plot_surface(func, resolution=10, show_contours=True)
        assert fig is not None

    def test_surface_custom_title_size(self):
        """Surface plot with custom title and size."""
        from surfaces._visualize import plot_surface

        func = AckleyFunction()
        fig = plot_surface(func, resolution=10, title="My Title", width=600, height=500)
        assert "My Title" in fig.layout.title.text
        assert fig.layout.width == 600
        assert fig.layout.height == 500


# =============================================================================
# Multi-Slice Plot Tests (Legacy Path)
# =============================================================================


@pytest.mark.viz
class TestMultiSliceLegacy:
    """Test plot_multi_slice with direct function + bounds (legacy path)."""

    def test_multi_slice_2d(self):
        """Multi-slice with 2D function."""
        from surfaces._visualize import plot_multi_slice

        func = AckleyFunction()
        fig = plot_multi_slice(func, resolution=10)
        assert fig is not None
        assert len(fig.data) == 2  # 2 dimensions = 2 traces

    def test_multi_slice_4d(self):
        """Multi-slice with 4D function."""
        from surfaces._visualize import plot_multi_slice

        func = SphereFunction(n_dim=4)
        fig = plot_multi_slice(func, resolution=10)
        assert fig is not None
        assert len(fig.data) == 4  # 4 dimensions = 4 traces

    def test_multi_slice_with_center(self):
        """Multi-slice with custom center point."""
        from surfaces._visualize import plot_multi_slice

        func = AckleyFunction()
        fig = plot_multi_slice(func, resolution=10, center={"x0": 1.0, "x1": 1.0})
        assert fig is not None

    def test_multi_slice_center_as_list(self):
        """Multi-slice with center as list."""
        from surfaces._visualize import plot_multi_slice

        func = AckleyFunction()
        fig = plot_multi_slice(func, resolution=10, center=[1.0, 1.0])
        assert fig is not None


# =============================================================================
# Auto-Plot Tests
# =============================================================================


@pytest.mark.viz
class TestAutoPlot:
    """Test auto_plot selection logic."""

    def test_auto_2d_gives_surface(self):
        """2D function auto-selects surface plot."""
        from surfaces._visualize import auto_plot

        func = AckleyFunction()
        fig = auto_plot(func, resolution=10)
        assert fig is not None
        # Surface plot has a Surface trace
        assert any("Surface" in str(type(trace)) for trace in fig.data)

    def test_auto_4d_gives_multi_slice(self):
        """4D function auto-selects multi-slice plot."""
        from surfaces._visualize import auto_plot

        func = SphereFunction(n_dim=4)
        fig = auto_plot(func, resolution=10)
        assert fig is not None
        # Multi-slice has Scatter traces
        assert any("Scatter" in str(type(trace)) for trace in fig.data)

    def test_auto_with_history_gives_convergence(self):
        """History argument triggers convergence plot."""
        from surfaces._visualize import auto_plot

        func = AckleyFunction()
        history = [10.0, 8.0, 5.0, 3.0, 1.0]
        fig = auto_plot(func, history=history)
        assert fig is not None


# =============================================================================
# Utils Tests
# =============================================================================


@pytest.mark.viz
class TestPlotUtils:
    """Test utility functions for visualization."""

    def test_create_search_space_grid_default(self):
        """Grid uses function's default bounds."""
        from surfaces._visualize._utils import create_search_space_grid

        func = AckleyFunction()
        grid = create_search_space_grid(func, resolution=10)
        assert "x0" in grid
        assert "x1" in grid
        assert len(grid["x0"]) == 10

    def test_create_search_space_grid_custom(self):
        """Grid with custom bounds."""
        from surfaces._visualize._utils import create_search_space_grid

        func = AckleyFunction()
        bounds = {"x0": (-1.0, 1.0), "x1": (-2.0, 2.0)}
        grid = create_search_space_grid(func, resolution=5, bounds=bounds)
        assert len(grid["x0"]) == 5
        assert grid["x0"][0] == pytest.approx(-1.0)
        assert grid["x0"][-1] == pytest.approx(1.0)

    def test_evaluate_grid_2d(self):
        """Evaluate function on 2D grid."""
        from surfaces._visualize._utils import evaluate_grid_2d

        func = SphereFunction(n_dim=2)
        x_vals = np.linspace(-1, 1, 5)
        y_vals = np.linspace(-1, 1, 5)
        z = evaluate_grid_2d(func, x_vals, y_vals, "x0", "x1")
        assert z.shape == (5, 5)
        # Center should be 0
        assert z[2, 2] == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_grid_2d_with_fixed(self):
        """Evaluate function on 2D grid with fixed params."""
        from surfaces._visualize._utils import evaluate_grid_2d

        func = SphereFunction(n_dim=3)
        x_vals = np.linspace(-1, 1, 5)
        y_vals = np.linspace(-1, 1, 5)
        z = evaluate_grid_2d(func, x_vals, y_vals, "x0", "x1", fixed_params={"x2": 0.0})
        assert z.shape == (5, 5)

    def test_validate_plot_incompatible(self):
        """validate_plot raises on incompatible function/plot."""
        from surfaces._visualize._errors import PlotCompatibilityError
        from surfaces._visualize._utils import validate_plot

        func = SphereFunction(n_dim=5)
        with pytest.raises(PlotCompatibilityError):
            validate_plot(func, "surface")  # surface requires 2D
