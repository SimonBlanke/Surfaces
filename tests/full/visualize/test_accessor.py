# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for PlotAccessor class.

These tests verify the accessor infrastructure.
They focus on:
- Accessor availability on test functions
- Method presence and signatures
"""

from surfaces._visualize._accessor import PlotAccessor
from surfaces.test_functions.algebraic.standard.test_functions_nd import SphereFunction


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
