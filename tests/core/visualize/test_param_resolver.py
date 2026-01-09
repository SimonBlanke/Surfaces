# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for parameter resolution (Phase 1).

These tests verify the params dict interpretation logic.
"""

import numpy as np
import pytest

from surfaces.test_functions.algebraic.standard.test_functions_nd import SphereFunction
from surfaces.visualize._param_resolver import (
    DimensionConfig,
    ResolvedParams,
    resolve_params,
)


class TestResolvedParamsDataclass:
    """Test ResolvedParams dataclass."""

    def test_empty_resolved_params(self):
        """Empty ResolvedParams has correct defaults."""
        resolved = ResolvedParams()
        assert resolved.plot_dims == []
        assert resolved.fixed_dims == []
        assert resolved.all_dims == []
        assert resolved.plot_dim_names == []
        assert resolved.fixed_dim_names == []
        assert resolved.fixed_values == {}

    def test_resolved_params_with_dims(self):
        """ResolvedParams correctly organizes dimensions."""
        plot_dim = DimensionConfig(
            name="x0",
            values=np.linspace(-5, 5, 10),
            is_plotted=True,
            bounds=(-5, 5),
        )
        fixed_dim = DimensionConfig(
            name="x1",
            values=np.array([0.0]),
            is_plotted=False,
            bounds=(0.0, 0.0),
        )

        resolved = ResolvedParams(
            plot_dims=[plot_dim],
            fixed_dims=[fixed_dim],
        )

        assert resolved.plot_dim_names == ["x0"]
        assert resolved.fixed_dim_names == ["x1"]
        assert resolved.fixed_values == {"x1": 0.0}

    def test_get_dim_by_name(self):
        """get_dim returns correct dimension config."""
        plot_dim = DimensionConfig(
            name="x0",
            values=np.linspace(-5, 5, 10),
            is_plotted=True,
            bounds=(-5, 5),
        )

        resolved = ResolvedParams(plot_dims=[plot_dim])

        assert resolved.get_dim("x0") is plot_dim
        assert resolved.get_dim("nonexistent") is None


class TestResolveParamsDefaults:
    """Test resolve_params with default values."""

    def test_none_params_uses_defaults(self):
        """None params uses all defaults."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(func, params=None, required_plot_dims=2)

        assert len(resolved.plot_dims) == 2
        assert len(resolved.fixed_dims) == 1
        assert resolved.plot_dim_names == ["x0", "x1"]
        assert resolved.fixed_dim_names == ["x2"]

    def test_empty_params_uses_defaults(self):
        """Empty params dict uses all defaults."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(func, params={}, required_plot_dims=2)

        assert resolved.plot_dim_names == ["x0", "x1"]

    def test_default_fixed_value_is_center(self):
        """Fixed dimensions use middle value from search_space by default."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(func, params=None, required_plot_dims=2)

        # x2 should be fixed at middle value from search_space (close to 0)
        assert resolved.fixed_values["x2"] == pytest.approx(0.0, abs=0.5)


class TestResolveParamsExplicit:
    """Test resolve_params with explicit values."""

    def test_ellipsis_marks_dimension_for_plotting(self):
        """Ellipsis (...) marks dimension for plotting with defaults."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(
            func,
            params={"x0": ..., "x2": ...},
            required_plot_dims=2,
        )

        assert resolved.plot_dim_names == ["x0", "x2"]
        assert "x1" in resolved.fixed_dim_names

    def test_tuple_marks_dimension_for_plotting(self):
        """Tuple (min, max) marks dimension for plotting."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(
            func,
            params={"x0": (-2, 2), "x1": (-1, 1)},
            required_plot_dims=2,
        )

        assert resolved.plot_dim_names == ["x0", "x1"]
        x0_dim = resolved.get_dim("x0")
        assert x0_dim.bounds == (-2, 2)

    def test_single_value_fixes_dimension(self):
        """Single numeric value fixes the dimension."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(
            func,
            params={"x0": ..., "x1": ..., "x2": 1.5},
            required_plot_dims=2,
        )

        assert "x2" in resolved.fixed_dim_names
        assert resolved.fixed_values["x2"] == 1.5

    def test_list_marks_dimension_for_plotting(self):
        """List of values marks dimension for plotting."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(
            func,
            params={"x0": [1, 2, 3, 4, 5], "x1": ...},
            required_plot_dims=2,
        )

        assert "x0" in resolved.plot_dim_names
        x0_dim = resolved.get_dim("x0")
        np.testing.assert_array_equal(x0_dim.values, [1, 2, 3, 4, 5])

    def test_range_marks_dimension_for_plotting(self):
        """range() object marks dimension for plotting."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(
            func,
            params={"x0": range(-2, 3), "x1": ...},
            required_plot_dims=2,
        )

        assert "x0" in resolved.plot_dim_names
        x0_dim = resolved.get_dim("x0")
        np.testing.assert_array_equal(x0_dim.values, [-2, -1, 0, 1, 2])


class TestResolveParamsValidation:
    """Test resolve_params validation."""

    def test_unknown_dimension_raises_error(self):
        """Unknown dimension name raises ValueError."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(ValueError, match="Unknown dimension"):
            resolve_params(func, params={"invalid": ...}, required_plot_dims=2)

    def test_wrong_plot_dim_count_raises_error(self):
        """Wrong number of plotted dimensions raises ValueError."""
        func = SphereFunction(n_dim=3)
        with pytest.raises(ValueError, match="exactly 2"):
            resolve_params(
                func,
                params={"x0": ...},  # Only 1 dimension marked for plotting
                required_plot_dims=2,
            )

    def test_too_many_plot_dims_raises_error(self):
        """Too many plotted dimensions raises ValueError."""
        func = SphereFunction(n_dim=3)
        with pytest.raises(ValueError, match="exactly 2"):
            resolve_params(
                func,
                params={"x0": ..., "x1": ..., "x2": ...},  # 3 dimensions
                required_plot_dims=2,
            )


class TestResolveParamsInference:
    """Test dimension inference from partial params."""

    def test_infer_plot_dims_from_fixed(self):
        """Infer plot dims when only fixed dims are specified."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(
            func,
            params={"x1": 0.5},  # Only x1 is fixed explicitly
            required_plot_dims=2,
        )

        # Should infer x0 and x2 as plot dims (first 2 non-fixed)
        assert len(resolved.plot_dims) == 2
        assert "x1" in resolved.fixed_dim_names
        assert resolved.fixed_values["x1"] == 0.5


class TestResolveParamsResolution:
    """Test resolution parameter for grid generation."""

    def test_resolution_affects_linspace(self):
        """Resolution parameter affects number of points in linspace."""
        func = SphereFunction(n_dim=2)
        resolved = resolve_params(
            func,
            params={"x0": (-2, 2), "x1": (-1, 1)},
            required_plot_dims=2,
            resolution=25,
        )

        x0_dim = resolved.get_dim("x0")
        assert len(x0_dim.values) == 25
