# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Auto-selection of best visualization for a test function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ..test_functions._base_test_function import BaseTestFunction

from ._compatibility import _get_function_dimensions
from ._convergence import plot_convergence
from ._slices import plot_multi_slice
from ._surface import plot_surface
from ._utils import check_plotly


def auto_plot(
    func: "BaseTestFunction",
    history: Optional[Union[List[float], Dict[str, List[float]]]] = None,
    resolution: int = 50,
    **kwargs,
) -> "go.Figure":
    """Automatically select and create the best visualization for a function.

    Selection logic:
    - 2D functions: surface plot
    - N-D functions (N > 2): multi_slice plot
    - 1D functions: multi_slice plot (single panel)
    - If history provided: convergence plot

    Args:
        func: A test function of any dimension.
        history: Optional optimization history. If provided, creates convergence plot.
        resolution: Resolution for grid-based plots.
        **kwargs: Additional arguments passed to the selected plot function.

    Returns:
        Plotly Figure object.

    Examples:
        >>> from surfaces.test_functions import SphereFunction
        >>> from surfaces.visualize import auto_plot
        >>> func = SphereFunction(n_dim=2)
        >>> fig = auto_plot(func)  # Returns surface plot
        >>> fig.show()
        >>> func5d = SphereFunction(n_dim=5)
        >>> fig = auto_plot(func5d)  # Returns multi_slice plot
        >>> fig.show()
    """
    check_plotly()

    # If history is provided, show convergence
    if history is not None:
        return plot_convergence(func, history, **kwargs)

    n_dim = _get_function_dimensions(func)

    # Select best plot for this function
    if n_dim == 2:
        return plot_surface(func, resolution=resolution, **kwargs)
    else:
        return plot_multi_slice(func, resolution=resolution, **kwargs)
