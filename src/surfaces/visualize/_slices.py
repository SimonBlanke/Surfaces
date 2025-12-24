# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Multi-slice plot for N-dimensional test functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ..test_functions._base_test_function import BaseTestFunction

from ._compatibility import _get_function_dimensions
from ._utils import (
    check_plotly,
    create_search_space_grid,
    go,
    make_subplots,
    validate_plot,
)


def plot_multi_slice(
    func: "BaseTestFunction",
    center: Optional[Union[Dict[str, float], List[float]]] = None,
    resolution: int = 100,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    title: Optional[str] = None,
    width: int = 900,
    height: Optional[int] = None,
) -> "go.Figure":
    """Create 1D slice plots through each dimension.

    Shows how the objective value changes along each parameter axis
    while keeping other parameters fixed at the center point.

    Args:
        func: A test function of any dimension.
        center: Center point for slicing. Defaults to zeros or middle of bounds.
        resolution: Number of points per slice (default: 100).
        bounds: Optional custom bounds per dimension.
        title: Plot title. Defaults to function name.
        width: Plot width in pixels.
        height: Plot height in pixels. Defaults to 200 per dimension.

    Returns:
        Plotly Figure with subplots for each dimension.

    Examples:
        >>> from surfaces.test_functions import SphereFunction
        >>> from surfaces.visualize import plot_multi_slice
        >>> func = SphereFunction(n_dim=5)
        >>> fig = plot_multi_slice(func)
        >>> fig.show()
    """
    check_plotly()
    validate_plot(func, "multi_slice")

    n_dim = _get_function_dimensions(func)
    search_space = create_search_space_grid(func, resolution, bounds)
    param_names = list(search_space.keys())

    # Determine center point
    if center is None:
        center = {name: 0.0 for name in param_names}
    elif isinstance(center, (list, tuple, np.ndarray)):
        center = {name: center[i] for i, name in enumerate(param_names)}

    # Create subplots
    height = height or 200 * n_dim
    fig = make_subplots(
        rows=n_dim,
        cols=1,
        subplot_titles=[f"Slice along {name}" for name in param_names],
        vertical_spacing=0.1 / n_dim if n_dim > 1 else 0.1,
    )

    for i, param_name in enumerate(param_names):
        x_values = search_space[param_name]
        y_values = []

        for x in x_values:
            params = center.copy()
            params[param_name] = x
            y_values.append(func(params))

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=param_name,
                line=dict(width=2),
            ),
            row=i + 1,
            col=1,
        )

        fig.update_xaxes(title_text=param_name, row=i + 1, col=1)
        fig.update_yaxes(title_text="Objective", row=i + 1, col=1)

    func_name = getattr(func, "name", type(func).__name__)
    fig.update_layout(
        title=title or f"{func_name} - Dimension Slices (center: {list(center.values())})",
        width=width,
        height=height,
        showlegend=False,
    )

    return fig
