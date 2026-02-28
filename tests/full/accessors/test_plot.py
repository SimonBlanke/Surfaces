"""Tests for PlotAccessor integration (requires plotly)."""

from surfaces.test_functions.algebraic import SphereFunction


class TestPlotAccessor:
    """Test PlotAccessor on concrete functions."""

    def test_plot_accessor_available(self):
        """Plot accessor is accessible."""
        func = SphereFunction(n_dim=2)
        assert hasattr(func.plot, "surface")
        assert hasattr(func.plot, "contour")
        assert hasattr(func.plot, "available")

    def test_plot_creates_figure(self):
        """Plot methods create plotly figures."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.surface()
        assert fig is not None
        assert type(fig).__name__ == "Figure"

    def test_plot_contour(self):
        """Contour plot works."""
        func = SphereFunction(n_dim=2)
        fig = func.plot.contour()
        assert fig is not None

    def test_plot_not_cached(self):
        """PlotAccessor is NOT cached (fresh each call)."""
        func = SphereFunction(n_dim=2)
        assert func.plot is not func.plot

    def test_plot_available_lists_methods(self):
        """available() returns list of plot method names."""
        func = SphereFunction(n_dim=2)
        available = func.plot.available()
        assert isinstance(available, list)
        assert "surface" in available
        assert "contour" in available
