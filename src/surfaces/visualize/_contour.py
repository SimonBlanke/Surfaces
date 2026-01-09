# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""2D contour and heatmap plots for test functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ..test_functions._base_test_function import BaseTestFunction
    from ._param_resolver import ResolvedParams

from ._utils import (
    DEFAULT_COLORSCALE,
    check_plotly,
    create_search_space_grid,
    evaluate_grid_2d,
    go,
    validate_plot,
)


def plot_contour(
    func: "BaseTestFunction",
    resolution: int = 50,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    title: Optional[str] = None,
    width: int = 700,
    height: int = 600,
    colorscale: Optional[str] = None,
    n_contours: int = 20,
    show_labels: bool = True,
    resolved: Optional["ResolvedParams"] = None,
    **kwargs: Any,
) -> "go.Figure":
    """Create a 2D contour plot of a 2D objective function.

    Args:
        func: A 2-dimensional test function, or N-dimensional with 2 plotted dims.
        resolution: Number of points per dimension (default: 50).
        bounds: Optional custom bounds as {'x0': (min, max), 'x1': (min, max)}.
            Ignored if resolved is provided.
        title: Plot title. Defaults to function name.
        width: Plot width in pixels.
        height: Plot height in pixels.
        colorscale: Plotly colorscale name.
        n_contours: Number of contour levels.
        show_labels: Whether to show contour value labels.
        resolved: Pre-resolved parameter configuration from PlotAccessor.
            If provided, bounds are ignored.
        **kwargs: Additional keyword arguments for future compatibility.

    Returns:
        Plotly Figure object.

    Raises:
        PlotCompatibilityError: If function doesn't have exactly 2 plotted dimensions.

    Examples:
        >>> from surfaces.test_functions import RosenbrockFunction
        >>> from surfaces.visualize import plot_contour
        >>> func = RosenbrockFunction(n_dim=2)
        >>> fig = plot_contour(func)
        >>> fig.show()

        # Using the accessor pattern:
        >>> fig = func.plot.contour()
    """
    check_plotly()

    # Use resolved params if provided (from accessor)
    if resolved is not None:
        return _plot_contour_from_resolved(
            func, resolved, title, width, height, colorscale, n_contours, show_labels
        )

    # Legacy path
    validate_plot(func, "contour")

    # Create grid
    search_space = create_search_space_grid(func, resolution, bounds)
    param_names = list(search_space.keys())[:2]
    x_name, y_name = param_names[0], param_names[1]

    x_values = search_space[x_name]
    y_values = search_space[y_name]

    # Evaluate
    z_values = evaluate_grid_2d(func, x_values, y_values, x_name, y_name)

    # Create figure
    fig = go.Figure(
        data=go.Contour(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale=colorscale or DEFAULT_COLORSCALE,
            contours=dict(
                showlabels=show_labels,
                labelfont=dict(size=10, color="white"),
            ),
            ncontours=n_contours,
        )
    )

    func_name = getattr(func, "name", type(func).__name__)
    fig.update_layout(
        title=title or f"{func_name} - Contour Plot",
        xaxis_title=x_name,
        yaxis_title=y_name,
        width=width,
        height=height,
    )

    return fig


def _plot_contour_from_resolved(
    func: "BaseTestFunction",
    resolved: "ResolvedParams",
    title: Optional[str] = None,
    width: int = 700,
    height: int = 600,
    colorscale: Optional[str] = None,
    n_contours: int = 20,
    show_labels: bool = True,
) -> "go.Figure":
    """Create contour plot from resolved parameters."""
    if len(resolved.plot_dims) != 2:
        raise ValueError(
            f"Contour plot requires exactly 2 plotted dimensions, "
            f"got {len(resolved.plot_dims)}: {resolved.plot_dim_names}"
        )

    x_dim, y_dim = resolved.plot_dims[0], resolved.plot_dims[1]
    x_values = x_dim.values
    y_values = y_dim.values

    # Get fixed values for other dimensions
    fixed_params = resolved.fixed_values

    # Evaluate
    z_values = evaluate_grid_2d(func, x_values, y_values, x_dim.name, y_dim.name, fixed_params)

    # Create figure
    fig = go.Figure(
        data=go.Contour(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale=colorscale or DEFAULT_COLORSCALE,
            contours=dict(
                showlabels=show_labels,
                labelfont=dict(size=10, color="white"),
            ),
            ncontours=n_contours,
        )
    )

    func_name = getattr(func, "name", type(func).__name__)
    default_title = f"{func_name} - Contour Plot"
    if fixed_params:
        fixed_str = ", ".join(f"{k}={v:.2f}" for k, v in fixed_params.items())
        default_title = f"{func_name} - Contour Plot (fixed: {fixed_str})"

    fig.update_layout(
        title=title or default_title,
        xaxis_title=x_dim.name,
        yaxis_title=y_dim.name,
        width=width,
        height=height,
    )

    return fig


def plot_heatmap(
    func: "BaseTestFunction",
    resolution: int = 50,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    title: Optional[str] = None,
    width: int = 700,
    height: int = 600,
    colorscale: Optional[str] = None,
    resolved: Optional["ResolvedParams"] = None,
    **kwargs: Any,
) -> "go.Figure":
    """Create a 2D heatmap plot of a 2D objective function.

    Args:
        func: A 2-dimensional test function, or N-dimensional with 2 plotted dims.
        resolution: Number of points per dimension (default: 50).
        bounds: Optional custom bounds as {'x0': (min, max), 'x1': (min, max)}.
            Ignored if resolved is provided.
        title: Plot title. Defaults to function name.
        width: Plot width in pixels.
        height: Plot height in pixels.
        colorscale: Plotly colorscale name.
        resolved: Pre-resolved parameter configuration from PlotAccessor.
            If provided, bounds are ignored.
        **kwargs: Additional keyword arguments for future compatibility.

    Returns:
        Plotly Figure object.

    Raises:
        PlotCompatibilityError: If function doesn't have exactly 2 plotted dimensions.

    Examples:
        >>> from surfaces.test_functions import AckleyFunction
        >>> from surfaces.visualize import plot_heatmap
        >>> func = AckleyFunction()
        >>> fig = plot_heatmap(func)
        >>> fig.show()

        # Using the accessor pattern:
        >>> fig = func.plot.heatmap()
    """
    check_plotly()

    # Use resolved params if provided (from accessor)
    if resolved is not None:
        return _plot_heatmap_from_resolved(func, resolved, title, width, height, colorscale)

    # Legacy path
    validate_plot(func, "heatmap")

    # Create grid
    search_space = create_search_space_grid(func, resolution, bounds)
    param_names = list(search_space.keys())[:2]
    x_name, y_name = param_names[0], param_names[1]

    x_values = search_space[x_name]
    y_values = search_space[y_name]

    # Evaluate
    z_values = evaluate_grid_2d(func, x_values, y_values, x_name, y_name)

    # Create figure
    fig = go.Figure(
        data=go.Heatmap(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale=colorscale or DEFAULT_COLORSCALE,
        )
    )

    func_name = getattr(func, "name", type(func).__name__)
    fig.update_layout(
        title=title or f"{func_name} - Heatmap",
        xaxis_title=x_name,
        yaxis_title=y_name,
        width=width,
        height=height,
    )

    return fig


def _plot_heatmap_from_resolved(
    func: "BaseTestFunction",
    resolved: "ResolvedParams",
    title: Optional[str] = None,
    width: int = 700,
    height: int = 600,
    colorscale: Optional[str] = None,
) -> "go.Figure":
    """Create heatmap plot from resolved parameters."""
    if len(resolved.plot_dims) != 2:
        raise ValueError(
            f"Heatmap plot requires exactly 2 plotted dimensions, "
            f"got {len(resolved.plot_dims)}: {resolved.plot_dim_names}"
        )

    x_dim, y_dim = resolved.plot_dims[0], resolved.plot_dims[1]
    x_values = x_dim.values
    y_values = y_dim.values

    # Get fixed values for other dimensions
    fixed_params = resolved.fixed_values

    # Evaluate
    z_values = evaluate_grid_2d(func, x_values, y_values, x_dim.name, y_dim.name, fixed_params)

    # Create figure
    fig = go.Figure(
        data=go.Heatmap(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale=colorscale or DEFAULT_COLORSCALE,
        )
    )

    func_name = getattr(func, "name", type(func).__name__)
    default_title = f"{func_name} - Heatmap"
    if fixed_params:
        fixed_str = ", ".join(f"{k}={v:.2f}" for k, v in fixed_params.items())
        default_title = f"{func_name} - Heatmap (fixed: {fixed_str})"

    fig.update_layout(
        title=title or default_title,
        xaxis_title=x_dim.name,
        yaxis_title=y_dim.name,
        width=width,
        height=height,
    )

    return fig
