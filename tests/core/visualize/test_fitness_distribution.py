# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for fitness_distribution plot via PlotAccessor.

These tests verify the fitness_distribution functionality through the accessor pattern.
"""

import numpy as np
import pytest

from surfaces.test_functions.algebraic.standard.test_functions_1d import (
    ForresterFunction,
)
from surfaces.test_functions.algebraic.standard.test_functions_2d import AckleyFunction
from surfaces.test_functions.algebraic.standard.test_functions_nd import (
    RastriginFunction,
    SphereFunction,
)
from surfaces.visualize._param_resolver import resolve_params


class TestFitnessDistributionBasic:
    """Basic functionality tests for fitness_distribution."""

    def test_1d_function_no_params(self):
        """Test fitness_distribution works with 1D function without params."""
        func = ForresterFunction()
        fig = func.plot.fitness_distribution(n_samples=50)

        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) > 0

    def test_2d_function_no_params(self):
        """Test fitness_distribution works with 2D function without params."""
        func = AckleyFunction()
        fig = func.plot.fitness_distribution(n_samples=50)

        assert fig is not None
        assert hasattr(fig, "data")

    def test_nd_function_no_params(self):
        """Test fitness_distribution works with N-dimensional function."""
        func = SphereFunction(n_dim=5)
        fig = func.plot.fitness_distribution(n_samples=50)

        assert fig is not None
        assert hasattr(fig, "data")

    def test_n_samples_parameter(self):
        """Test that n_samples parameter is respected."""
        func = SphereFunction(n_dim=2)

        # Reset function to clear search data
        func.reset()

        # Use small n_samples and check function was called that many times
        fig = func.plot.fitness_distribution(n_samples=25)

        # The function should have been called 25 times
        assert func.n_evaluations == 25

    def test_returns_plotly_figure(self):
        """Test that the method returns a Plotly Figure."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.fitness_distribution(n_samples=10)

        # Check it's a Plotly Figure
        assert type(fig).__name__ == "Figure"
        assert hasattr(fig, "update_layout")
        assert hasattr(fig, "add_trace")


class TestFitnessDistributionParamResolution:
    """Test parameter resolution for fitness_distribution."""

    def test_no_params_samples_all_dims(self):
        """Without params, all dimensions should be sampled."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(
            func, params=None, required_plot_dims=None, plot_all_by_default=True
        )

        # All dimensions should be plotted (sampled)
        assert len(resolved.plot_dims) == 3
        assert len(resolved.fixed_dims) == 0

    def test_fixed_dim_uses_fixed_value(self):
        """Fixed dimensions should use their fixed value, not be sampled."""
        func = SphereFunction(n_dim=3)
        resolved = resolve_params(
            func,
            params={"x1": 0.5},  # Fix x1 to 0.5
            required_plot_dims=None,
            plot_all_by_default=True,
        )

        assert len(resolved.plot_dims) == 2  # x0 and x2 sampled
        assert len(resolved.fixed_dims) == 1  # x1 fixed
        assert resolved.fixed_values["x1"] == 0.5

    def test_multiple_fixed_dims(self):
        """Multiple fixed dimensions should all use their fixed values."""
        func = SphereFunction(n_dim=5)
        resolved = resolve_params(
            func,
            params={"x0": 0.0, "x2": 1.0, "x4": -1.0},
            required_plot_dims=None,
            plot_all_by_default=True,
        )

        assert len(resolved.plot_dims) == 2  # x1 and x3 sampled
        assert len(resolved.fixed_dims) == 3
        assert resolved.fixed_values["x0"] == 0.0
        assert resolved.fixed_values["x2"] == 1.0
        assert resolved.fixed_values["x4"] == -1.0


class TestFitnessDistributionWithParams:
    """Test fitness_distribution with various params configurations."""

    def test_with_custom_range(self):
        """Test that custom range restricts sampling bounds."""
        func = SphereFunction(n_dim=2)
        func.reset()

        # Use a small range to verify sampling is within bounds
        fig = func.plot.fitness_distribution(
            params={"x0": (-1, 1), "x1": (-1, 1)},
            n_samples=100,
        )

        # Check that all samples were within bounds
        # by verifying the function values are reasonable for the restricted range
        assert fig is not None
        assert func.n_evaluations == 100

        # For Sphere function with |x| <= 1, max value is 2 (1^2 + 1^2)
        max_score = max(record["score"] for record in func.search_data)
        assert max_score <= 2.0 + 0.01  # Small tolerance

    def test_with_fixed_dimensions(self):
        """Test fitness_distribution with some dimensions fixed."""
        func = SphereFunction(n_dim=3)
        func.reset()

        # Fix x2 to 0, sample x0 and x1
        fig = func.plot.fitness_distribution(
            params={"x2": 0.0},
            n_samples=50,
        )

        assert fig is not None
        assert func.n_evaluations == 50

        # All evaluations should have x2 = 0
        for record in func.search_data:
            assert record["x2"] == 0.0

    def test_with_ellipsis_and_fixed(self):
        """Test mixing ellipsis (sample) and fixed values."""
        func = SphereFunction(n_dim=4)
        func.reset()

        # Explicitly sample x0 and x2, fix x1 and x3
        fig = func.plot.fitness_distribution(
            params={
                "x0": ...,
                "x1": 0.5,
                "x2": ...,
                "x3": -0.5,
            },
            n_samples=30,
        )

        assert fig is not None
        assert func.n_evaluations == 30

        # Check fixed dimensions
        for record in func.search_data:
            assert record["x1"] == 0.5
            assert record["x3"] == -0.5


class TestFitnessDistributionStats:
    """Test statistics display in fitness_distribution."""

    def test_show_stats_true(self):
        """Test that statistics are shown when show_stats=True."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.fitness_distribution(n_samples=50, show_stats=True)

        # Check that an annotation was added
        assert len(fig.layout.annotations) > 0

        # Check annotation contains expected statistics
        annotation_text = fig.layout.annotations[0].text
        assert "Mean" in annotation_text
        assert "Std" in annotation_text
        assert "Min" in annotation_text
        assert "Max" in annotation_text

    def test_show_stats_false(self):
        """Test that statistics are hidden when show_stats=False."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.fitness_distribution(n_samples=50, show_stats=False)

        # No statistics annotation should be present
        assert len(fig.layout.annotations) == 0


class TestFitnessDistributionTitle:
    """Test title handling in fitness_distribution."""

    def test_default_title(self):
        """Test that default title includes function name and sample count."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.fitness_distribution(n_samples=100)

        assert "Sphere" in fig.layout.title.text
        assert "100" in fig.layout.title.text

    def test_custom_title(self):
        """Test that custom title is used when provided."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.fitness_distribution(n_samples=50, title="My Custom Title")

        assert fig.layout.title.text == "My Custom Title"


class TestFitnessDistributionDimensions:
    """Test fitness_distribution with various dimension configurations."""

    def test_high_dimensional_function(self):
        """Test fitness_distribution works with high-dimensional functions."""
        func = SphereFunction(n_dim=10)
        fig = func.plot.fitness_distribution(n_samples=20)

        assert fig is not None

    def test_only_one_dim_sampled(self):
        """Test with all but one dimension fixed."""
        func = SphereFunction(n_dim=3)
        func.reset()

        fig = func.plot.fitness_distribution(
            params={
                "x0": ...,  # Only x0 sampled
                "x1": 0.0,
                "x2": 0.0,
            },
            n_samples=30,
        )

        assert fig is not None
        assert func.n_evaluations == 30

        # x1 and x2 should be fixed
        for record in func.search_data:
            assert record["x1"] == 0.0
            assert record["x2"] == 0.0


class TestFitnessDistributionReproducibility:
    """Test reproducibility of fitness_distribution."""

    def test_reproducible_results(self):
        """Test that results are reproducible (seeded random)."""
        func1 = SphereFunction(n_dim=2)
        func2 = SphereFunction(n_dim=2)

        func1.reset()
        func2.reset()

        fig1 = func1.plot.fitness_distribution(n_samples=50)
        fig2 = func2.plot.fitness_distribution(n_samples=50)

        # Both should have the same samples (same seed)
        scores1 = [record["score"] for record in func1.search_data]
        scores2 = [record["score"] for record in func2.search_data]

        assert scores1 == scores2
