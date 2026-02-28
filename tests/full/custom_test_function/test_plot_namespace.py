# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CustomTestFunction plot namespace (requires matplotlib)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from surfaces.custom_test_function import CustomTestFunction


def sphere(params):
    return sum(v**2 for v in params.values())


@pytest.fixture
def sphere_func():
    """Simple sphere function for testing."""
    return CustomTestFunction(
        objective_fn=sphere,
        search_space={"x": (-5, 5), "y": (-5, 5)},
    )


@pytest.fixture
def sphere_func_with_data(sphere_func):
    """Sphere function with 50 evaluation data points."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        x, y = rng.uniform(-5, 5, 2)
        sphere_func({"x": x, "y": y})
    return sphere_func


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test."""
    yield
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass


class TestPlotHistory:
    """Test plot.history() method."""

    def test_history_returns_axes(self, sphere_func_with_data):
        """history() returns matplotlib Axes."""
        import matplotlib.axes

        ax = sphere_func_with_data.plot.history()
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_history_show_best_false(self, sphere_func_with_data):
        """history() works with show_best=False."""
        ax = sphere_func_with_data.plot.history(show_best=False)
        assert ax is not None

    def test_history_log_scale(self, sphere_func_with_data):
        """history() works with log_scale=True."""
        ax = sphere_func_with_data.plot.history(log_scale=True)
        assert ax.get_yscale() == "log"

    def test_history_raises_no_data(self, sphere_func):
        """history() raises with no evaluation data."""
        with pytest.raises(ValueError, match="at least 1 evaluations"):
            sphere_func.plot.history()


class TestPlotImportance:
    """Test plot.importance() method."""

    def test_importance_returns_axes(self, sphere_func_with_data):
        """importance() returns matplotlib Axes."""
        import matplotlib.axes

        ax = sphere_func_with_data.plot.importance()
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_importance_raises_insufficient_data(self, sphere_func):
        """importance() raises with insufficient data."""
        # Add just a few evaluations (less than 10)
        for i in range(3):
            sphere_func({"x": float(i), "y": float(i)})
        with pytest.raises(ValueError, match="at least 10 evaluations"):
            sphere_func.plot.importance()


class TestPlotContour:
    """Test plot.contour() method."""

    def test_contour_returns_axes(self, sphere_func_with_data):
        """contour() returns matplotlib Axes."""
        import matplotlib.axes

        ax = sphere_func_with_data.plot.contour("x", "y")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_contour_unknown_param_raises(self, sphere_func_with_data):
        """contour() raises for unknown parameter name."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            sphere_func_with_data.plot.contour("x", "z")

    def test_contour_insufficient_data_raises(self, sphere_func):
        """contour() raises with insufficient data."""
        sphere_func({"x": 1.0, "y": 1.0})
        with pytest.raises(ValueError, match="at least 10 evaluations"):
            sphere_func.plot.contour("x", "y")


class TestPlotParallelCoordinates:
    """Test plot.parallel_coordinates() method."""

    def test_parallel_coordinates_returns_axes(self, sphere_func_with_data):
        """parallel_coordinates() returns matplotlib Axes."""
        import matplotlib.axes

        ax = sphere_func_with_data.plot.parallel_coordinates()
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_parallel_coordinates_top_k(self, sphere_func_with_data):
        """parallel_coordinates() works with top_k filter."""
        ax = sphere_func_with_data.plot.parallel_coordinates(top_k=10)
        assert ax is not None

    def test_parallel_coordinates_insufficient_data(self, sphere_func):
        """parallel_coordinates() raises with insufficient data."""
        sphere_func({"x": 1.0, "y": 1.0})
        with pytest.raises(ValueError, match="at least 5 evaluations"):
            sphere_func.plot.parallel_coordinates()
