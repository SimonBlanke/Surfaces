# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""3D surface plot for 2D test functions."""

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


def plot_surface(
    func: "BaseTestFunction",
    resolution: int = 50,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 700,
    colorscale: Optional[str] = None,
    show_contours: bool = False,
    resolved: Optional["ResolvedParams"] = None,
    **kwargs: Any,
) -> "go.Figure":
    """Create a 3D surface plot of a 2D objective function.

    Args:
        func: A 2-dimensional test function, or N-dimensional with 2 plotted dims.
        resolution: Number of points per dimension (default: 50).
        bounds: Optional custom bounds as {'x0': (min, max), 'x1': (min, max)}.
            Ignored if resolved is provided.
        title: Plot title. Defaults to function name.
        width: Plot width in pixels.
        height: Plot height in pixels.
        colorscale: Plotly colorscale name.
        show_contours: Whether to show contour projection on the bottom.
        resolved: Pre-resolved parameter configuration from PlotAccessor.
            If provided, bounds are ignored.
        **kwargs: Additional keyword arguments for future compatibility.

    Returns:
        Plotly Figure object.

    Raises:
        PlotCompatibilityError: If function doesn't have exactly 2 plotted dimensions.

    Examples:
        >>> from surfaces.test_functions import AckleyFunction
        >>> from surfaces.visualize import plot_surface
        >>> func = AckleyFunction()
        >>> fig = plot_surface(func, resolution=100)
        >>> fig.show()

        # Using the accessor pattern:
        >>> fig = func.plot.surface()
    """
    check_plotly()

    # Use resolved params if provided (from accessor)
    if resolved is not None:
        return _plot_surface_from_resolved(
            func, resolved, title, width, height, colorscale, show_contours
        )

    # Legacy path
    validate_plot(func, "surface")

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
        data=go.Surface(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale=colorscale or DEFAULT_COLORSCALE,
            contours_z=dict(
                show=show_contours,
                usecolormap=True,
                project_z=show_contours,
            )
            if show_contours
            else None,
        )
    )

    func_name = getattr(func, "name", type(func).__name__)
    fig.update_layout(
        title=title or f"{func_name} - Surface Plot",
        scene=dict(
            xaxis_title=x_name,
            yaxis_title=y_name,
            zaxis_title="Objective Value",
        ),
        width=width,
        height=height,
    )

    return fig


def _plot_surface_from_resolved(
    func: "BaseTestFunction",
    resolved: "ResolvedParams",
    title: Optional[str] = None,
    width: int = 800,
    height: int = 700,
    colorscale: Optional[str] = None,
    show_contours: bool = False,
) -> "go.Figure":
    """Create surface plot from resolved parameters.

    This is the new implementation path used by the PlotAccessor.
    """
    if len(resolved.plot_dims) != 2:
        raise ValueError(
            f"Surface plot requires exactly 2 plotted dimensions, "
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
        data=go.Surface(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale=colorscale or DEFAULT_COLORSCALE,
            contours_z=dict(
                show=show_contours,
                usecolormap=True,
                project_z=show_contours,
            )
            if show_contours
            else None,
        )
    )

    func_name = getattr(func, "name", type(func).__name__)
    default_title = f"{func_name} - Surface Plot"
    if fixed_params:
        fixed_str = ", ".join(f"{k}={v:.2f}" for k, v in fixed_params.items())
        default_title = f"{func_name} - Surface Plot (fixed: {fixed_str})"

    fig.update_layout(
        title=title or default_title,
        scene=dict(
            xaxis_title=x_dim.name,
            yaxis_title=y_dim.name,
            zaxis_title="Objective Value",
        ),
        width=width,
        height=height,
    )

    return fig
