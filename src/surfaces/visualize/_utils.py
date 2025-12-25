# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Shared utilities for visualization plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..test_functions._base_test_function import BaseTestFunction

# Check for visualization dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
    DEFAULT_COLORSCALE = px.colors.sequential.Viridis
except ImportError:
    HAS_PLOTLY = False
    go = None
    px = None
    make_subplots = None
    DEFAULT_COLORSCALE = None

from ._compatibility import _get_function_dimensions, check_compatibility
from ._errors import (
    MissingDependencyError,
    PlotCompatibilityError,
    get_alternative_suggestions,
)


def check_plotly() -> None:
    """Check if plotly is available, raise helpful error if not."""
    if not HAS_PLOTLY:
        raise MissingDependencyError(["plotly"])


def validate_plot(
    func: "BaseTestFunction",
    plot_name: str,
    has_history: bool = False,
) -> None:
    """Validate that a plot is compatible with a function.

    Raises PlotCompatibilityError with suggestions if incompatible.
    """
    is_compatible, reason = check_compatibility(func, plot_name, has_history)

    if not is_compatible:
        suggestions = get_alternative_suggestions(plot_name, func)
        raise PlotCompatibilityError(
            plot_name=plot_name,
            reason=reason,
            func=func,
            suggestions=suggestions,
        )


def create_search_space_grid(
    func: "BaseTestFunction",
    resolution: int = 50,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, np.ndarray]:
    """Create a grid search space for visualization.

    Args:
        func: The test function.
        resolution: Number of points per dimension.
        bounds: Optional custom bounds per dimension.

    Returns:
        Dictionary mapping dimension names to arrays of values.
    """
    if bounds is not None:
        return {name: np.linspace(b[0], b[1], resolution) for name, b in bounds.items()}

    # Use function's default bounds
    default_bounds = getattr(func, "default_bounds", (-5.0, 5.0))
    n_dim = _get_function_dimensions(func)

    return {
        f"x{i}": np.linspace(default_bounds[0], default_bounds[1], resolution) for i in range(n_dim)
    }


def evaluate_grid_2d(
    func: "BaseTestFunction",
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_name: str = "x0",
    y_name: str = "x1",
    fixed_params: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Evaluate function on a 2D grid.

    Args:
        func: The test function.
        x_values: Values for the x dimension.
        y_values: Values for the y dimension.
        x_name: Name of the x dimension.
        y_name: Name of the y dimension.
        fixed_params: Fixed values for other dimensions.

    Returns:
        2D array of function values.
    """
    fixed_params = fixed_params or {}
    z_values = np.zeros((len(y_values), len(x_values)))

    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            params = {**fixed_params, x_name: x, y_name: y}
            z_values[i, j] = func(params)

    return z_values
