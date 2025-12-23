# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""2D contour plot for 2D test functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ..test_functions._base_test_function import BaseTestFunction

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
) -> "go.Figure":
    """Create a 2D contour plot of a 2D objective function.

    Args:
        func: A 2-dimensional test function.
        resolution: Number of points per dimension (default: 50).
        bounds: Optional custom bounds as {'x0': (min, max), 'x1': (min, max)}.
        title: Plot title. Defaults to function name.
        width: Plot width in pixels.
        height: Plot height in pixels.
        colorscale: Plotly colorscale name.
        n_contours: Number of contour levels.
        show_labels: Whether to show contour value labels.

    Returns:
        Plotly Figure object.

    Raises:
        PlotCompatibilityError: If function is not 2-dimensional.

    Examples:
        >>> from surfaces.test_functions import RosenbrockFunction
        >>> from surfaces.visualize import plot_contour
        >>> func = RosenbrockFunction(n_dim=2)
        >>> fig = plot_contour(func)
        >>> fig.show()
    """
    check_plotly()
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
