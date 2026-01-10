# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for convergence plot via PlotAccessor.

These tests verify the convergence functionality through the accessor pattern.
"""

import numpy as np
import pytest

from surfaces.test_functions.algebraic import SphereFunction


class TestConvergenceBasic:
    """Basic functionality tests for convergence."""

    def test_with_search_data(self):
        """Test convergence works with function's search_data."""
        func = SphereFunction(n_dim=2)

        # Run some evaluations
        for i in range(10):
            func([i * 0.1, i * 0.1])

        fig = func.plot.convergence()

        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) > 0

    def test_with_explicit_history_list(self):
        """Test convergence with explicit history as list of floats."""
        func = SphereFunction(n_dim=2)
        history = [10.0, 8.0, 5.0, 3.0, 1.0, 0.5, 0.1]

        fig = func.plot.convergence(history=history)

        assert fig is not None
        assert len(fig.data) == 1  # Single run

    def test_with_explicit_history_dict(self):
        """Test convergence with explicit history as dict of runs."""
        func = SphereFunction(n_dim=2)
        history = {
            "Run 1": [10, 8, 5, 3, 1],
            "Run 2": [12, 9, 6, 4, 2],
        }

        fig = func.plot.convergence(history=history)

        assert fig is not None
        assert len(fig.data) == 2  # Two runs

    def test_with_history_numpy_1d(self):
        """Test convergence with 1D numpy array."""
        func = SphereFunction(n_dim=2)
        history = np.array([10.0, 8.0, 5.0, 3.0, 1.0])

        fig = func.plot.convergence(history=history)

        assert fig is not None
        assert len(fig.data) == 1

    def test_with_history_numpy_2d(self):
        """Test convergence with 2D numpy array (multiple runs)."""
        func = SphereFunction(n_dim=2)
        history = np.array(
            [
                [10, 8, 5, 3, 1],
                [12, 9, 6, 4, 2],
                [15, 10, 7, 5, 3],
            ]
        )

        fig = func.plot.convergence(history=history)

        assert fig is not None
        assert len(fig.data) == 3  # Three runs

    def test_returns_plotly_figure(self):
        """Test that the method returns a Plotly Figure."""
        func = SphereFunction(n_dim=2)
        history = [10, 8, 5, 3, 1]

        fig = func.plot.convergence(history=history)

        assert type(fig).__name__ == "Figure"
        assert hasattr(fig, "update_layout")


class TestConvergenceWithHistory:
    """Test with_history chaining."""

    def test_with_history_method(self):
        """Test using with_history to set history."""
        func = SphereFunction(n_dim=2)
        history = [10, 8, 5, 3, 1]

        fig = func.plot.with_history(history).convergence()

        assert fig is not None

    def test_with_history_returns_accessor(self):
        """Test that with_history returns PlotAccessor for chaining."""
        func = SphereFunction(n_dim=2)
        history = [10, 8, 5, 3, 1]

        accessor = func.plot.with_history(history)

        # Should be a PlotAccessor
        assert hasattr(accessor, "convergence")
        assert hasattr(accessor, "surface")

    def test_explicit_history_overrides_with_history(self):
        """Test that explicit history parameter overrides with_history."""
        func = SphereFunction(n_dim=2)
        history1 = [10, 8, 5]  # via with_history
        history2 = [20, 15, 10, 5, 1]  # explicit

        fig = func.plot.with_history(history1).convergence(history=history2)

        # Should use history2 (5 points, not 3)
        x_data = fig.data[0].x
        assert len(x_data) == 5


class TestConvergenceSearchDataFormat:
    """Test convergence with search_data format (list of dicts)."""

    def test_handles_search_data_dicts(self):
        """Test that search_data format (list of dicts) is handled."""
        func = SphereFunction(n_dim=2)

        # search_data is list of dicts with 'score' key
        func([0.0, 0.0])  # score = 0
        func([1.0, 1.0])  # score = 2
        func([0.5, 0.5])  # score = 0.5

        fig = func.plot.convergence()

        assert fig is not None
        # Should have extracted scores correctly
        y_data = list(fig.data[0].y)
        assert len(y_data) == 3

    def test_handles_explicit_search_data_format(self):
        """Test with explicitly passed search_data format."""
        func = SphereFunction(n_dim=2)
        history = [
            {"x0": 0.0, "x1": 0.0, "score": 10.0},
            {"x0": 0.5, "x1": 0.5, "score": 5.0},
            {"x0": 1.0, "x1": 1.0, "score": 2.0},
        ]

        fig = func.plot.convergence(history=history)

        assert fig is not None
        y_data = list(fig.data[0].y)
        # With show_best=True (default), should be cumulative min
        assert y_data == [10.0, 5.0, 2.0]


class TestConvergenceErrors:
    """Test error handling in convergence."""

    def test_error_without_history(self):
        """Test that error is raised when no history is available."""
        func = SphereFunction(n_dim=2)
        # No evaluations, no explicit history

        with pytest.raises(ValueError, match="No history data available"):
            func.plot.convergence()

    def test_error_with_empty_history(self):
        """Test that error is raised with empty history."""
        func = SphereFunction(n_dim=2)

        with pytest.raises(ValueError, match="No history data available"):
            func.plot.convergence(history=[])


class TestConvergenceShowBest:
    """Test show_best parameter."""

    def test_show_best_true_cumulative_min(self):
        """Test that show_best=True shows cumulative minimum."""
        func = SphereFunction(n_dim=2)
        history = [10, 5, 8, 3, 6, 1]  # Not monotonic

        fig = func.plot.convergence(history=history, show_best=True)

        y_data = list(fig.data[0].y)
        # Cumulative minimum: [10, 5, 5, 3, 3, 1]
        assert y_data == [10, 5, 5, 3, 3, 1]

    def test_show_best_false_raw_values(self):
        """Test that show_best=False shows raw values."""
        func = SphereFunction(n_dim=2)
        history = [10, 5, 8, 3, 6, 1]

        fig = func.plot.convergence(history=history, show_best=False)

        y_data = list(fig.data[0].y)
        # Raw values unchanged
        assert y_data == [10, 5, 8, 3, 6, 1]


class TestConvergenceLogScale:
    """Test log_scale parameter."""

    def test_log_scale_true(self):
        """Test that log_scale=True sets y-axis to log."""
        func = SphereFunction(n_dim=2)
        history = [100, 10, 1, 0.1, 0.01]

        fig = func.plot.convergence(history=history, log_scale=True)

        assert fig.layout.yaxis.type == "log"

    def test_log_scale_false(self):
        """Test that log_scale=False keeps linear y-axis."""
        func = SphereFunction(n_dim=2)
        history = [100, 10, 1]

        fig = func.plot.convergence(history=history, log_scale=False)

        # yaxis.type should be None or 'linear' (default)
        assert fig.layout.yaxis.type in [None, "linear", "-"]


class TestConvergenceTitle:
    """Test title handling."""

    def test_default_title(self):
        """Test that default title includes function name."""
        func = SphereFunction(n_dim=2)
        history = [10, 5, 1]

        fig = func.plot.convergence(history=history)

        assert "Sphere" in fig.layout.title.text
        assert "Convergence" in fig.layout.title.text

    def test_custom_title(self):
        """Test that custom title is used."""
        func = SphereFunction(n_dim=2)
        history = [10, 5, 1]

        fig = func.plot.convergence(history=history, title="My Custom Title")

        assert fig.layout.title.text == "My Custom Title"


class TestConvergenceAxisLabels:
    """Test axis labels."""

    def test_x_axis_label(self):
        """Test that x-axis is labeled 'Evaluation'."""
        func = SphereFunction(n_dim=2)
        history = [10, 5, 1]

        fig = func.plot.convergence(history=history)

        assert fig.layout.xaxis.title.text == "Evaluation"

    def test_y_axis_label_show_best_true(self):
        """Test y-axis label with show_best=True."""
        func = SphereFunction(n_dim=2)
        history = [10, 5, 1]

        fig = func.plot.convergence(history=history, show_best=True)

        assert "Best" in fig.layout.yaxis.title.text

    def test_y_axis_label_show_best_false(self):
        """Test y-axis label with show_best=False."""
        func = SphereFunction(n_dim=2)
        history = [10, 5, 1]

        fig = func.plot.convergence(history=history, show_best=False)

        assert fig.layout.yaxis.title.text == "Objective Value"


class TestConvergenceMultipleRuns:
    """Test convergence with multiple optimization runs."""

    def test_multiple_runs_dict(self):
        """Test multiple runs via dict."""
        func = SphereFunction(n_dim=2)
        history = {
            "Algorithm A": [10, 5, 2, 1],
            "Algorithm B": [12, 8, 4, 2],
        }

        fig = func.plot.convergence(history=history)

        # Should have 2 traces
        assert len(fig.data) == 2
        # Check trace names
        trace_names = [trace.name for trace in fig.data]
        assert "Algorithm A" in trace_names
        assert "Algorithm B" in trace_names

    def test_multiple_runs_different_lengths(self):
        """Test multiple runs with different lengths."""
        func = SphereFunction(n_dim=2)
        history = {
            "Short": [10, 5],
            "Long": [12, 8, 4, 2, 1],
        }

        fig = func.plot.convergence(history=history)

        assert len(fig.data) == 2
        # Check lengths
        assert len(fig.data[0].x) == 2
        assert len(fig.data[1].x) == 5
