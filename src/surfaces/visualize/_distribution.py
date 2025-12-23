# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Fitness distribution plot for test functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ..test_functions._base_test_function import BaseTestFunction

from ._compatibility import _get_function_dimensions
from ._utils import check_plotly, go, validate_plot


def plot_fitness_distribution(
    func: "BaseTestFunction",
    n_samples: int = 10000,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    title: Optional[str] = None,
    width: int = 700,
    height: int = 500,
    n_bins: int = 50,
    show_stats: bool = True,
) -> "go.Figure":
    """Create a histogram of objective values from random sampling.

    Helps understand the distribution of function values across the search space.

    Args:
        func: A test function of any dimension.
        n_samples: Number of random samples to evaluate (default: 10000).
        bounds: Optional custom bounds per dimension.
        title: Plot title. Defaults to function name.
        width: Plot width in pixels.
        height: Plot height in pixels.
        n_bins: Number of histogram bins.
        show_stats: Whether to show mean, std, min, max annotations.

    Returns:
        Plotly Figure object.

    Examples:
        >>> from surfaces.test_functions import RastriginFunction
        >>> from surfaces.visualize import plot_fitness_distribution
        >>> func = RastriginFunction(n_dim=10)
        >>> fig = plot_fitness_distribution(func, n_samples=5000)
        >>> fig.show()
    """
    check_plotly()
    validate_plot(func, "fitness_distribution")

    n_dim = _get_function_dimensions(func)

    # Determine bounds
    if bounds is None:
        default_bounds = getattr(func, "default_bounds", (-5.0, 5.0))
        bounds = {f"x{i}": default_bounds for i in range(n_dim)}

    # Generate random samples
    np.random.seed(42)  # Reproducibility
    samples = []
    for _ in range(n_samples):
        params = {}
        for name, (low, high) in bounds.items():
            params[name] = np.random.uniform(low, high)
        samples.append(func(params))

    samples = np.array(samples)

    # Create histogram
    fig = go.Figure(
        data=go.Histogram(
            x=samples,
            nbinsx=n_bins,
            marker_color="steelblue",
            opacity=0.75,
        )
    )

    func_name = getattr(func, "name", type(func).__name__)
    fig.update_layout(
        title=title or f"{func_name} - Fitness Distribution ({n_samples} samples)",
        xaxis_title="Objective Value",
        yaxis_title="Count",
        width=width,
        height=height,
    )

    if show_stats:
        stats_text = (
            f"Mean: {np.mean(samples):.4f}<br>"
            f"Std: {np.std(samples):.4f}<br>"
            f"Min: {np.min(samples):.4f}<br>"
            f"Max: {np.max(samples):.4f}"
        )
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        )

    return fig
