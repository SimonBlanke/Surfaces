# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Multi-slice plot for N-dimensional test functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ..test_functions._base_test_function import BaseTestFunction
    from ._param_resolver import ResolvedParams

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
    resolved: Optional["ResolvedParams"] = None,
    **kwargs: Any,
) -> "go.Figure":
    """Create 1D slice plots through each dimension.

    Shows how the objective value changes along each parameter axis
    while keeping other parameters fixed at the center point.

    Args:
        func: A test function of any dimension.
        center: Center point for slicing. Defaults to zeros or middle of bounds.
            Ignored if resolved is provided.
        resolution: Number of points per slice (default: 100).
        bounds: Optional custom bounds per dimension.
            Ignored if resolved is provided.
        title: Plot title. Defaults to function name.
        width: Plot width in pixels.
        height: Plot height in pixels. Defaults to 200 per dimension.
        resolved: Pre-resolved parameter configuration from PlotAccessor.
            If provided, center and bounds are ignored.
        **kwargs: Additional keyword arguments for future compatibility.

    Returns:
        Plotly Figure with subplots for each dimension.

    Examples:
        >>> from surfaces.test_functions import SphereFunction
        >>> from surfaces.visualize import plot_multi_slice
        >>> func = SphereFunction(n_dim=5)
        >>> fig = plot_multi_slice(func)
        >>> fig.show()

        # Using the accessor pattern:
        >>> fig = func.plot.multi_slice()
    """
    check_plotly()
    validate_plot(func, "multi_slice")

    # Use resolved params if provided (from accessor)
    if resolved is not None:
        return _plot_multi_slice_from_resolved(func, resolved, title, width, height)

    # Legacy path: use center/bounds directly
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


def _plot_multi_slice_from_resolved(
    func: "BaseTestFunction",
    resolved: "ResolvedParams",
    title: Optional[str] = None,
    width: int = 900,
    height: Optional[int] = None,
) -> "go.Figure":
    """Create multi-slice plot from resolved parameters.

    This is the new implementation path used by the PlotAccessor.
    """
    # Get dimensions to plot (all plotted dims get their own slice)
    plot_dims = resolved.plot_dims
    n_plots = len(plot_dims)

    if n_plots == 0:
        raise ValueError("No dimensions configured for plotting in multi_slice")

    # Build center point from all dimensions
    # For plot dims: use their center
    # For fixed dims: use their fixed value
    center = {}
    for dim in resolved.plot_dims:
        center[dim.name] = dim.center
    for dim in resolved.fixed_dims:
        center[dim.name] = dim.values[0]

    # Create subplots
    height = height or 200 * n_plots
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        subplot_titles=[f"Slice along {dim.name}" for dim in plot_dims],
        vertical_spacing=0.1 / n_plots if n_plots > 1 else 0.1,
    )

    for i, dim in enumerate(plot_dims):
        x_values = dim.values
        y_values = []

        for x in x_values:
            params = center.copy()
            params[dim.name] = x
            y_values.append(func(params))

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=dim.name,
                line=dict(width=2),
            ),
            row=i + 1,
            col=1,
        )

        fig.update_xaxes(title_text=dim.name, row=i + 1, col=1)
        fig.update_yaxes(title_text="Objective", row=i + 1, col=1)

    # Build title showing center values
    center_values = [f"{center[dim.name]:.2f}" for dim in plot_dims]
    func_name = getattr(func, "name", type(func).__name__)
    default_title = f"{func_name} - Dimension Slices (center: [{', '.join(center_values)}])"

    fig.update_layout(
        title=title or default_title,
        width=width,
        height=height,
        showlegend=False,
    )

    return fig
