# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for Plotly visualization functions.

These tests verify plot structure and data without rendering.
Plotly figures are dictionaries under the hood, so we can inspect
the data structure directly.
"""

import numpy as np
import pytest

from surfaces._visualize import (
    _create_grid,
    _plot_parameter_slice,
    _plotly_contour,
    _plotly_heatmap,
    _plotly_surface,
    _plotly_surface_nd,
)
from surfaces.test_functions.algebraic import algebraic_functions_2d

from tests.conftest import func_id, instantiate_function


# =============================================================================
# Grid Creation Tests
# =============================================================================


@pytest.mark.viz
class TestGridCreation:
    """Tests for grid creation helper."""

    def test_grid_shape(self):
        """Grid has correct shape based on search space resolution."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 15),
        }

        xi, yi, zi = _create_grid(simple_func, search_space)

        assert xi.shape == (15, 10)
        assert yi.shape == (15, 10)
        assert zi.shape == (15, 10)

    def test_grid_values(self):
        """Grid computes correct function values."""

        def sphere_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.array([-1, 0, 1]),
            "y": np.array([-1, 0, 1]),
        }

        xi, yi, zi = _create_grid(sphere_func, search_space)

        # At origin (0, 0) the value should be 0
        assert zi[1, 1] == pytest.approx(0.0)
        # At corners the value should be 2 (1^2 + 1^2)
        assert zi[0, 0] == pytest.approx(2.0)
        assert zi[2, 2] == pytest.approx(2.0)


# =============================================================================
# Surface Plot Tests
# =============================================================================


@pytest.mark.viz
class TestPlotlySurface:
    """Tests for 3D surface plot generation."""

    def test_surface_returns_figure(self):
        """Surface function returns a Plotly figure object."""
        import plotly.graph_objects as go

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _plotly_surface(simple_func, search_space)

        assert isinstance(fig, go.Figure)

    def test_surface_data_structure(self):
        """Surface plot has correct data structure."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _plotly_surface(simple_func, search_space)
        fig_dict = fig.to_dict()

        # Check data structure
        assert len(fig_dict["data"]) == 1
        assert fig_dict["data"][0]["type"] == "surface"
        assert "x" in fig_dict["data"][0]
        assert "y" in fig_dict["data"][0]
        assert "z" in fig_dict["data"][0]

    def test_surface_z_data_shape(self):
        """Surface z-data has correct dimensions."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        resolution = 15
        search_space = {
            "x": np.linspace(-5, 5, resolution),
            "y": np.linspace(-5, 5, resolution),
        }

        fig = _plotly_surface(simple_func, search_space)

        z_data = np.array(fig.data[0].z)
        assert z_data.shape == (resolution, resolution)

    def test_surface_layout(self):
        """Surface plot has correct layout settings."""

        def simple_func(params):
            return params["x"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        title = "Test Surface"
        width = 800
        height = 600

        fig = _plotly_surface(simple_func, search_space, title=title, width=width, height=height)
        layout = fig.to_dict()["layout"]

        assert layout["title"]["text"] == title
        assert layout["width"] == width
        assert layout["height"] == height

    def test_surface_axis_labels(self):
        """Surface plot has correct axis labels from search space keys."""

        def simple_func(params):
            return params["alpha"] + params["beta"]

        search_space = {
            "alpha": np.linspace(0, 1, 5),
            "beta": np.linspace(0, 1, 5),
        }

        fig = _plotly_surface(simple_func, search_space)
        scene = fig.to_dict()["layout"]["scene"]

        assert scene["xaxis"]["title"]["text"] == "alpha"
        assert scene["yaxis"]["title"]["text"] == "beta"
        assert scene["zaxis"]["title"]["text"] == "Metric"

    def test_surface_with_contour(self):
        """Surface plot can include contour projections."""

        def simple_func(params):
            return params["x"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _plotly_surface(simple_func, search_space, contour=True)
        trace = fig.to_dict()["data"][0]

        assert "contours" in trace
        assert trace["contours"]["z"]["show"] is True

    def test_surface_rejects_1d_space(self):
        """Surface plot raises error for 1D search space."""

        def simple_func(params):
            return params["x"] ** 2

        search_space = {"x": np.linspace(-5, 5, 10)}

        with pytest.raises(ValueError, match="two dimensional"):
            _plotly_surface(simple_func, search_space)

    @pytest.mark.parametrize("func_class", algebraic_functions_2d[:5], ids=func_id)
    def test_surface_with_real_functions(self, func_class):
        """Surface plot works with actual test functions."""
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
        fig = _plotly_surface(func, reduced_space)

        assert fig.data[0].type == "surface"
        z_data = np.array(fig.data[0].z)
        assert z_data.shape == (10, 10)
        assert np.all(np.isfinite(z_data))


# =============================================================================
# Heatmap Tests
# =============================================================================


@pytest.mark.viz
class TestPlotlyHeatmap:
    """Tests for 2D heatmap generation."""

    def test_heatmap_returns_figure(self):
        """Heatmap function returns a Plotly figure object."""
        import plotly.graph_objects as go

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _plotly_heatmap(simple_func, search_space)

        assert isinstance(fig, go.Figure)

    def test_heatmap_data_structure(self):
        """Heatmap has correct data structure."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _plotly_heatmap(simple_func, search_space)
        fig_dict = fig.to_dict()

        assert len(fig_dict["data"]) == 1
        # px.imshow creates a heatmap trace
        assert fig_dict["data"][0]["type"] == "heatmap"

    def test_heatmap_dimensions(self):
        """Heatmap z-data has correct dimensions."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        x_res, y_res = 12, 8
        search_space = {
            "x": np.linspace(-5, 5, x_res),
            "y": np.linspace(-5, 5, y_res),
        }

        fig = _plotly_heatmap(simple_func, search_space)

        z_data = np.array(fig.data[0].z)
        assert z_data.shape == (y_res, x_res)

    def test_heatmap_layout(self):
        """Heatmap has correct layout settings."""

        def simple_func(params):
            return params["x"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        title = "Test Heatmap"
        fig = _plotly_heatmap(simple_func, search_space, title=title)
        layout = fig.to_dict()["layout"]

        assert layout["title"]["text"] == title

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
            _plotly_heatmap(simple_func, search_space)


# =============================================================================
# Contour Plot Tests
# =============================================================================


@pytest.mark.viz
class TestPlotlyContour:
    """Tests for contour plot generation."""

    def test_contour_returns_figure(self):
        """Contour function returns a Plotly figure object."""
        import plotly.graph_objects as go

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _plotly_contour(simple_func, search_space)

        assert isinstance(fig, go.Figure)

    def test_contour_data_structure(self):
        """Contour plot has correct data structure."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _plotly_contour(simple_func, search_space)
        fig_dict = fig.to_dict()

        assert len(fig_dict["data"]) == 1
        assert fig_dict["data"][0]["type"] == "contour"

    def test_contour_levels(self):
        """Contour plot respects contour_levels parameter."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 20),
            "y": np.linspace(-5, 5, 20),
        }

        levels = 15
        fig = _plotly_contour(simple_func, search_space, contour_levels=levels)
        contours = fig.to_dict()["data"][0]["contours"]

        # Check that contour size is set based on levels
        assert "size" in contours


# =============================================================================
# N-Dimensional Surface Tests
# =============================================================================


@pytest.mark.viz
class TestPlotlySurfaceND:
    """Tests for N-dimensional surface plot (projects to 2D)."""

    def test_nd_surface_with_extra_dims(self):
        """N-D surface handles dimensions > 2 by fixing extras."""

        def nd_func(params):
            return sum(v**2 for v in params.values())

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
            "z": np.array([0.0]),  # Fixed dimension
        }

        fig = _plotly_surface_nd(nd_func, search_space)

        assert fig.data[0].type == "surface"
        z_data = np.array(fig.data[0].z)
        assert z_data.shape == (10, 10)

    def test_nd_surface_rejects_1d(self):
        """N-D surface raises error for 1D search space."""

        def simple_func(params):
            return params["x"] ** 2

        search_space = {"x": np.linspace(-5, 5, 10)}

        with pytest.raises(ValueError, match="at least two dimensional"):
            _plotly_surface_nd(simple_func, search_space)


# =============================================================================
# Parameter Slice Tests
# =============================================================================


@pytest.mark.viz
class TestParameterSlice:
    """Tests for 1D parameter slice plots."""

    def test_slice_returns_figure(self):
        """Slice function returns a Plotly figure object."""
        import plotly.graph_objects as go

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _plot_parameter_slice(
            simple_func,
            search_space,
            slice_param="x",
            fixed_params={"y": 0.0},
        )

        assert isinstance(fig, go.Figure)

    def test_slice_data_structure(self):
        """Slice plot has correct data structure."""

        def simple_func(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {
            "x": np.linspace(-5, 5, 10),
            "y": np.linspace(-5, 5, 10),
        }

        fig = _plot_parameter_slice(
            simple_func,
            search_space,
            slice_param="x",
            fixed_params={"y": 0.0},
        )
        fig_dict = fig.to_dict()

        assert len(fig_dict["data"]) == 1
        assert fig_dict["data"][0]["type"] == "scatter"
        assert fig_dict["data"][0]["mode"] == "lines+markers"

    def test_slice_invalid_param(self):
        """Slice raises error for invalid parameter name."""

        def simple_func(params):
            return params["x"] ** 2

        search_space = {"x": np.linspace(-5, 5, 10)}

        with pytest.raises(ValueError, match="not found"):
            _plot_parameter_slice(
                simple_func,
                search_space,
                slice_param="invalid",
                fixed_params={},
            )

    def test_slice_values_correct(self):
        """Slice plot computes correct function values along parameter."""

        def simple_func(params):
            return params["x"] ** 2

        x_values = np.array([-2, -1, 0, 1, 2])
        search_space = {
            "x": x_values,
            "y": np.array([0.0]),
        }

        fig = _plot_parameter_slice(
            simple_func,
            search_space,
            slice_param="x",
            fixed_params={"y": 0.0},
        )

        y_data = np.array(fig.data[0].y)
        expected = x_values**2

        np.testing.assert_array_almost_equal(y_data, expected)
