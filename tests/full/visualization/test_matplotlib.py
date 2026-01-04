# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for Matplotlib visualization functions.

These tests verify plot structure and data without displaying.
Matplotlib figures can be inspected programmatically.
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pytest

from surfaces._visualize import _matplotlib_heatmap, _matplotlib_surface
from surfaces.test_functions.algebraic import algebraic_functions_2d

from tests.conftest import func_id, instantiate_function


# =============================================================================
# Matplotlib Heatmap Tests
# =============================================================================


@pytest.mark.viz
class TestMatplotlibHeatmap:
    """Tests for matplotlib heatmap generation."""

    def test_heatmap_returns_figure(self):
        """Heatmap function returns a matplotlib figure."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _matplotlib_heatmap(simple_func, search_space)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_has_axes(self):
        """Heatmap figure has exactly one axes."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _matplotlib_heatmap(simple_func, search_space)

        assert len(fig.axes) >= 1  # Main axes + colorbar
        plt.close(fig)

    def test_heatmap_axis_labels(self):
        """Heatmap has correct axis labels from search space keys."""

        def simple_func(params):
            return params["alpha"] ** 2 + params["beta"] ** 2

        search_space = {
            "alpha": np.linspace(-5, 5, 10),
            "beta": np.linspace(-5, 5, 10),
        }

        fig = _matplotlib_heatmap(simple_func, search_space)
        ax = fig.axes[0]

        assert ax.get_xlabel() == "alpha"
        assert ax.get_ylabel() == "beta"
        plt.close(fig)

    def test_heatmap_title(self):
        """Heatmap has correct title."""

        def simple_func(params):
            return params["x"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        title = "Test Heatmap Title"
        fig = _matplotlib_heatmap(simple_func, search_space, title=title)
        ax = fig.axes[0]

        assert ax.get_title() == title
        plt.close(fig)

    def test_heatmap_figsize(self):
        """Heatmap respects figsize parameter."""

        def simple_func(params):
            return params["x"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        figsize = (12, 8)
        fig = _matplotlib_heatmap(simple_func, search_space, figsize=figsize)

        # Get figure size in inches
        actual_size = fig.get_size_inches()
        np.testing.assert_array_almost_equal(actual_size, figsize, decimal=1)
        plt.close(fig)

    def test_heatmap_rejects_wrong_dimensions(self):
        """Heatmap raises error for non-2D search space."""

        def simple_func(params):
            return sum(params.values())

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
            "z": np.linspace(-5, 5, 10),
        }

        with pytest.raises(ValueError, match="two dimensional"):
            _matplotlib_heatmap(simple_func, search_space)

    def test_heatmap_image_data(self):
        """Heatmap image data has correct shape."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        x_res, y_res = 15, 20
        search_space = {
            "x": np.linspace(-5, 5, x_res),
            "y": np.linspace(-5, 5, y_res),
        }

        fig = _matplotlib_heatmap(simple_func, search_space)
        ax = fig.axes[0]

        # Get the image data from the axes
        images = ax.get_images()
        assert len(images) == 1

        image_data = images[0].get_array()
        assert image_data.shape == (y_res, x_res)
        plt.close(fig)

    @pytest.mark.parametrize("func_class", algebraic_functions_2d[:3], ids=func_id)
    def test_heatmap_with_real_functions(self, func_class):
        """Heatmap works with actual test functions."""
        func = instantiate_function(func_class)
        search_space = func.search_space

        # Use lower resolution for speed
        reduced_space = {}
        for k, v in search_space.items():
            if isinstance(v, np.ndarray):
                reduced_space[k] = np.linspace(float(v.min()), float(v.max()), 10)
            else:
                reduced_space[k] = v[:10]

        # Functions are directly callable via __call__
        fig = _matplotlib_heatmap(func, reduced_space)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)


# =============================================================================
# Matplotlib Surface Tests
# =============================================================================


@pytest.mark.viz
class TestMatplotlibSurface:
    """Tests for matplotlib 3D surface plot generation."""

    def test_surface_returns_figure(self):
        """Surface function returns a matplotlib figure."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _matplotlib_surface(simple_func, search_space)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_surface_has_3d_axes(self):
        """Surface figure has 3D axes."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _matplotlib_surface(simple_func, search_space)
        ax = fig.axes[0]

        assert hasattr(ax, "plot_surface")  # 3D axes have this method
        plt.close(fig)

    def test_surface_axis_labels(self):
        """Surface has correct axis labels."""

        def simple_func(params):
            return params["alpha"] ** 2 + params["beta"] ** 2

        search_space = {
            "alpha": np.linspace(-5, 5, 10),
            "beta": np.linspace(-5, 5, 10),
        }

        fig = _matplotlib_surface(simple_func, search_space)
        ax = fig.axes[0]

        assert ax.get_xlabel() == "alpha"
        assert ax.get_ylabel() == "beta"
        assert ax.get_zlabel() == "Metric"
        plt.close(fig)

    def test_surface_title(self):
        """Surface has correct title."""

        def simple_func(params):
            return params["x"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        title = "Test Surface Title"
        fig = _matplotlib_surface(simple_func, search_space, title=title)
        ax = fig.axes[0]

        assert ax.get_title() == title
        plt.close(fig)

    def test_surface_figsize(self):
        """Surface respects figsize parameter."""

        def simple_func(params):
            return params["x"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        figsize = (12, 10)
        fig = _matplotlib_surface(simple_func, search_space, figsize=figsize)

        actual_size = fig.get_size_inches()
        np.testing.assert_array_almost_equal(actual_size, figsize, decimal=1)
        plt.close(fig)

    def test_surface_rejects_wrong_dimensions(self):
        """Surface raises error for non-2D search space."""

        def simple_func(params):
            return sum(params.values())

        search_space = {"x": np.linspace(-5, 5, 10)}

        with pytest.raises(ValueError, match="two dimensional"):
            _matplotlib_surface(simple_func, search_space)

    @pytest.mark.parametrize("func_class", algebraic_functions_2d[:3], ids=func_id)
    def test_surface_with_real_functions(self, func_class):
        """Surface works with actual test functions."""
        func = instantiate_function(func_class)
        search_space = func.search_space

        # Use lower resolution for speed
        reduced_space = {}
        for k, v in search_space.items():
            if isinstance(v, np.ndarray):
                reduced_space[k] = np.linspace(float(v.min()), float(v.max()), 8)
            else:
                reduced_space[k] = v[:8]

        # Functions are directly callable via __call__
        fig = _matplotlib_surface(func, reduced_space)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Resource Cleanup Tests
# =============================================================================


@pytest.mark.viz
class TestResourceCleanup:
    """Tests to verify figures are properly cleaned up."""

    def test_no_figure_leak(self):
        """Creating figures doesn't leak memory when closed."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        initial_figs = plt.get_fignums()

        for _ in range(5):
            fig = _matplotlib_heatmap(simple_func, search_space)
            plt.close(fig)

        final_figs = plt.get_fignums()
        assert len(final_figs) == len(initial_figs)
