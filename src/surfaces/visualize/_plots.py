# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Plot implementations for test function visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ..test_functions._base_test_function import BaseTestFunction

# Check for visualization dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
    DEFAULT_COLORSCALE = px.colors.sequential.Viridis
except ImportError:
    _HAS_PLOTLY = False
    go = None
    px = None
    make_subplots = None
    DEFAULT_COLORSCALE = None


from ._compatibility import (
    PLOT_REGISTRY,
    _get_function_dimensions,
    check_compatibility,
)
from ._errors import (
    MissingDataError,
    MissingDependencyError,
    PlotCompatibilityError,
    get_alternative_suggestions,
)


def _check_plotly():
    """Check if plotly is available, raise helpful error if not."""
    if not _HAS_PLOTLY:
        raise MissingDependencyError(["plotly"])


def _validate_plot(
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


def _create_search_space_grid(
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
        return {
            name: np.linspace(b[0], b[1], resolution)
            for name, b in bounds.items()
        }

    # Use function's default bounds
    default_bounds = getattr(func, "default_bounds", (-5.0, 5.0))
    n_dim = _get_function_dimensions(func)

    return {
        f"x{i}": np.linspace(default_bounds[0], default_bounds[1], resolution)
        for i in range(n_dim)
    }


def _evaluate_grid_2d(
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


# =============================================================================
# Plot 1: 3D Surface Plot (2D functions only)
# =============================================================================


def plot_surface(
    func: "BaseTestFunction",
    resolution: int = 50,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 700,
    colorscale: Optional[str] = None,
    show_contours: bool = False,
) -> "go.Figure":
    """Create a 3D surface plot of a 2D objective function.

    Args:
        func: A 2-dimensional test function.
        resolution: Number of points per dimension (default: 50).
        bounds: Optional custom bounds as {'x0': (min, max), 'x1': (min, max)}.
        title: Plot title. Defaults to function name.
        width: Plot width in pixels.
        height: Plot height in pixels.
        colorscale: Plotly colorscale name.
        show_contours: Whether to show contour projection on the bottom.

    Returns:
        Plotly Figure object.

    Raises:
        PlotCompatibilityError: If function is not 2-dimensional.

    Examples:
        >>> from surfaces.test_functions import AckleyFunction
        >>> from surfaces.visualize import plot_surface
        >>> func = AckleyFunction()
        >>> fig = plot_surface(func, resolution=100)
        >>> fig.show()
    """
    _check_plotly()
    _validate_plot(func, "surface")

    # Create grid
    search_space = _create_search_space_grid(func, resolution, bounds)
    param_names = list(search_space.keys())[:2]
    x_name, y_name = param_names[0], param_names[1]

    x_values = search_space[x_name]
    y_values = search_space[y_name]

    # Evaluate
    z_values = _evaluate_grid_2d(func, x_values, y_values, x_name, y_name)

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


# =============================================================================
# Plot 2: Contour Plot (2D functions only)
# =============================================================================


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
    _check_plotly()
    _validate_plot(func, "contour")

    # Create grid
    search_space = _create_search_space_grid(func, resolution, bounds)
    param_names = list(search_space.keys())[:2]
    x_name, y_name = param_names[0], param_names[1]

    x_values = search_space[x_name]
    y_values = search_space[y_name]

    # Evaluate
    z_values = _evaluate_grid_2d(func, x_values, y_values, x_name, y_name)

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


# =============================================================================
# Plot 3: Multi-Slice Plot (any dimension)
# =============================================================================


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
    _check_plotly()
    _validate_plot(func, "multi_slice")

    n_dim = _get_function_dimensions(func)
    search_space = _create_search_space_grid(func, resolution, bounds)
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


# =============================================================================
# Plot 4: Convergence Plot (requires history)
# =============================================================================


def plot_convergence(
    func: "BaseTestFunction",
    history: Union[List[float], Dict[str, List[float]], np.ndarray],
    title: Optional[str] = None,
    width: int = 800,
    height: int = 500,
    log_scale: bool = False,
    show_best: bool = True,
) -> "go.Figure":
    """Create a convergence plot showing optimization progress.

    Args:
        func: The test function (used for title and context).
        history: Objective values per evaluation. Can be:
            - List of values from a single run
            - Dict mapping run names to lists of values
            - 2D array where each row is a run
        title: Plot title. Defaults to function name.
        width: Plot width in pixels.
        height: Plot height in pixels.
        log_scale: Whether to use log scale for y-axis.
        show_best: Whether to show best-so-far instead of raw values.

    Returns:
        Plotly Figure object.

    Raises:
        MissingDataError: If history is empty.

    Examples:
        >>> from surfaces.test_functions import SphereFunction
        >>> from surfaces.visualize import plot_convergence
        >>> func = SphereFunction(n_dim=2)
        >>> # Simulated optimization history
        >>> history = [10.0, 8.0, 5.0, 3.0, 1.0, 0.5, 0.1]
        >>> fig = plot_convergence(func, history)
        >>> fig.show()
    """
    _check_plotly()

    if history is None or (hasattr(history, "__len__") and len(history) == 0):
        raise MissingDataError("convergence", "optimization history")

    # Normalize history to dict format
    if isinstance(history, (list, np.ndarray)):
        if isinstance(history, np.ndarray) and history.ndim == 2:
            # Multiple runs as 2D array
            history = {f"Run {i+1}": list(row) for i, row in enumerate(history)}
        else:
            # Single run
            history = {"Optimization": list(history)}

    fig = go.Figure()

    for run_name, values in history.items():
        if show_best:
            # Cumulative minimum (best so far)
            best_so_far = np.minimum.accumulate(values)
            y_values = best_so_far
            y_label = "Best Objective Value"
        else:
            y_values = values
            y_label = "Objective Value"

        evaluations = list(range(1, len(values) + 1))

        fig.add_trace(
            go.Scatter(
                x=evaluations,
                y=y_values,
                mode="lines",
                name=run_name,
                line=dict(width=2),
            )
        )

    func_name = getattr(func, "name", type(func).__name__)
    fig.update_layout(
        title=title or f"{func_name} - Convergence",
        xaxis_title="Evaluation",
        yaxis_title=y_label,
        width=width,
        height=height,
    )

    if log_scale:
        fig.update_yaxes(type="log")

    return fig


# =============================================================================
# Plot 5: Fitness Distribution (any dimension)
# =============================================================================


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
    _check_plotly()
    _validate_plot(func, "fitness_distribution")

    n_dim = _get_function_dimensions(func)

    # Determine bounds
    if bounds is None:
        default_bounds = getattr(func, "default_bounds", (-5.0, 5.0))
        bounds = {
            f"x{i}": default_bounds for i in range(n_dim)
        }

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


# =============================================================================
# Auto Plot: Automatically select best visualization
# =============================================================================


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
    _check_plotly()

    # If history is provided, show convergence
    if history is not None:
        return plot_convergence(func, history, **kwargs)

    n_dim = _get_function_dimensions(func)

    # Select best plot for this function
    if n_dim == 2:
        return plot_surface(func, resolution=resolution, **kwargs)
    else:
        return plot_multi_slice(func, resolution=resolution, **kwargs)
