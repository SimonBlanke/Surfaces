# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for multi_slice plot via PlotAccessor.

These tests verify the multi_slice functionality through the accessor pattern.
"""

import pytest

from surfaces.test_functions.algebraic.standard.test_functions_nd import SphereFunction
from surfaces.visualize._param_resolver import resolve_params


class TestMultiSliceParamResolution:
    """Test parameter resolution for multi_slice (plot_all_by_default=True)."""

    def test_no_params_plots_all_dims(self):
        """Without params, all dimensions should be plotted."""
        func = SphereFunction(n_dim=5)
        resolved = resolve_params(
            func, params=None, required_plot_dims=None, plot_all_by_default=True
        )

        assert len(resolved.plot_dims) == 5
        assert len(resolved.fixed_dims) == 0
        assert resolved.plot_dim_names == ["x0", "x1", "x2", "x3", "x4"]

    def test_empty_params_plots_all_dims(self):
        """Empty params dict should plot all dimensions."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(
            func, params={}, required_plot_dims=None, plot_all_by_default=True
        )

        assert len(resolved.plot_dims) == 3
        assert resolved.plot_dim_names == ["x0", "x1", "x2"]

    def test_fixed_dim_excludes_from_plot(self):
        """Fixing a dimension should exclude it from plots."""
        func = SphereFunction(n_dim=4)
        resolved = resolve_params(
            func,
            params={"x1": 0.5},  # Fix x1
            required_plot_dims=None,
            plot_all_by_default=True,
        )

        assert len(resolved.plot_dims) == 3
        assert len(resolved.fixed_dims) == 1
        assert "x1" not in resolved.plot_dim_names
        assert "x1" in resolved.fixed_dim_names
        assert resolved.fixed_values["x1"] == 0.5

    def test_multiple_fixed_dims(self):
        """Multiple fixed dimensions should all be excluded."""
        func = SphereFunction(n_dim=5)
        resolved = resolve_params(
            func,
            params={"x0": 0.0, "x2": 1.0, "x4": -1.0},  # Fix 3 dims
            required_plot_dims=None,
            plot_all_by_default=True,
        )

        assert len(resolved.plot_dims) == 2
        assert len(resolved.fixed_dims) == 3
        assert resolved.plot_dim_names == ["x1", "x3"]

    def test_custom_range_for_dim(self):
        """Custom range marks only that dimension for plotting.

        Note: Specifying a tuple for one dimension means "plot this dimension".
        Other dimensions become fixed. Use ellipsis to keep them plotted.
        """
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(
            func,
            params={"x0": (-2, 2)},
            required_plot_dims=None,
            plot_all_by_default=True,
        )

        # Only x0 is explicitly marked, so only x0 is plotted
        assert len(resolved.plot_dims) == 1
        x0_dim = resolved.get_dim("x0")
        assert x0_dim.bounds == (-2, 2)

    def test_custom_range_with_ellipsis_keeps_all(self):
        """Custom range with ellipsis for other dims keeps all plotted."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(
            func,
            params={"x0": (-2, 2), "x1": ..., "x2": ...},
            required_plot_dims=None,
            plot_all_by_default=True,
        )

        # All 3 dims are explicitly marked for plotting
        assert len(resolved.plot_dims) == 3
        x0_dim = resolved.get_dim("x0")
        assert x0_dim.bounds == (-2, 2)

    def test_ellipsis_uses_defaults(self):
        """Ellipsis should use default bounds."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(
            func,
            params={"x0": ..., "x1": ...},  # Explicit ellipsis
            required_plot_dims=None,
            plot_all_by_default=True,
        )

        # When explicit plot markers are given, only those are plotted
        assert len(resolved.plot_dims) == 2
        assert resolved.plot_dim_names == ["x0", "x1"]


class TestDimensionConfigCenter:
    """Test DimensionConfig center property."""

    def test_center_is_midpoint(self):
        """Center should be midpoint of bounds."""
        func = SphereFunction(n_dim=2)
        resolved = resolve_params(
            func,
            params={"x0": (-4, 4)},
            required_plot_dims=None,
            plot_all_by_default=True,
        )

        x0_dim = resolved.get_dim("x0")
        assert x0_dim.center == 0.0  # Midpoint of -4 to 4

    def test_center_with_asymmetric_bounds(self):
        """Center calculation with asymmetric bounds."""
        func = SphereFunction(n_dim=2)
        resolved = resolve_params(
            func,
            params={"x0": (0, 10)},
            required_plot_dims=None,
            plot_all_by_default=True,
        )

        x0_dim = resolved.get_dim("x0")
        assert x0_dim.center == 5.0  # Midpoint of 0 to 10


class TestMultiSliceVisualization:
    """Test multi_slice plot creation via accessor.

    These tests require plotly to be installed. They test the full
    accessor -> param_resolver -> plot_multi_slice pipeline.
    """

    @pytest.fixture
    def plotly_available(self):
        """Check if plotly is available."""
        try:
            import plotly.graph_objects as go  # noqa: F401

            return True
        except ImportError:
            pytest.skip("plotly not installed")

    def test_multi_slice_no_params(self, plotly_available):
        """multi_slice() without params should show all dimensions."""
        func = SphereFunction(n_dim=3)
        fig = func.plot.multi_slice()

        assert fig is not None
        # Should have 3 traces (one per dimension)
        assert len(fig.data) == 3

    def test_multi_slice_with_custom_range(self, plotly_available):
        """multi_slice() with custom range for one dim plots only that dim."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.multi_slice(params={"x0": (-2, 2)})

        assert fig is not None
        # Only x0 is marked for plotting, so only 1 trace
        assert len(fig.data) == 1

    def test_multi_slice_with_custom_range_all_dims(self, plotly_available):
        """multi_slice() with custom range and ellipsis keeps all dims."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.multi_slice(params={"x0": (-2, 2), "x1": ...})

        assert fig is not None
        # Both dims marked for plotting
        assert len(fig.data) == 2

    def test_multi_slice_with_fixed_dim(self, plotly_available):
        """multi_slice() with one fixed dimension."""
        func = SphereFunction(n_dim=4)
        fig = func.plot.multi_slice(params={"x1": 0.0})

        assert fig is not None
        # Should have 3 traces (x0, x2, x3 - x1 is fixed)
        assert len(fig.data) == 3

    def test_multi_slice_with_resolution(self, plotly_available):
        """multi_slice() respects resolution parameter."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.multi_slice(resolution=25)

        assert fig is not None
        # Each trace should have 25 points
        assert len(fig.data[0].x) == 25
        assert len(fig.data[1].x) == 25

    def test_multi_slice_2d_function(self, plotly_available):
        """multi_slice() works with 2D function."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.multi_slice()

        assert fig is not None
        assert len(fig.data) == 2

    def test_multi_slice_5d_function(self, plotly_available):
        """multi_slice() works with 5D function."""
        func = SphereFunction(n_dim=5)
        fig = func.plot.multi_slice()

        assert fig is not None
        assert len(fig.data) == 5
