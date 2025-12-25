# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

# Internal module - not part of public API
# Visualization utilities for test functions (experimental)

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go

    _HAS_VIZ_DEPS = True
    color_scale = px.colors.sequential.Jet
except ImportError:
    _HAS_VIZ_DEPS = False
    go = None  # type: ignore[assignment]
    px = None
    mpl = None
    plt = None  # type: ignore[assignment]
    color_scale = None


def _check_viz_deps() -> None:
    """Check if visualization dependencies are available."""
    if not _HAS_VIZ_DEPS:
        raise ImportError(
            "Visualization features require matplotlib and plotly. "
            "Install with: pip install surfaces[viz]"
        )


def _create_grid(objective_function: Any, search_space: Dict[str, np.ndarray]) -> tuple:
    """Create a 2D grid for visualization from a search space and objective function.

    Args:
        objective_function: Function that takes a dict of parameters and returns a scalar
        search_space: Dictionary with exactly 2 keys, each mapping to numpy arrays

    Returns:
        tuple: (xi, yi, zi) meshgrid arrays for plotting
    """
    _check_viz_deps()

    def objective_function_np(*args: Any) -> float:
        para = {}
        for arg, key in zip(args, search_space.keys()):
            para[key] = arg
        return objective_function(para)

    (x_all, y_all) = search_space.values()
    xi, yi = np.meshgrid(x_all, y_all)
    zi = objective_function_np(xi, yi)

    return xi, yi, zi


def _plotly_surface_nd(
    objective_function,
    search_space: Dict[str, np.ndarray],
    title: str = "Objective Function Surface",
    width: int = 900,
    height: int = 900,
    contour: bool = False,
) -> go.Figure:
    """Create a 3D surface plot from an N-dimensional objective function.

    For N > 2 dimensions, automatically reduces to 2D by:
    1. Using single-value dimensions as fixed parameters
    2. Taking first two multi-value dimensions for visualization

    Args:
        objective_function: Function that takes a dict of parameters
        search_space: Dictionary mapping parameter names to value arrays
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels
        contour: Whether to add contour projections

    Returns:
        Plotly Figure object

    Raises:
        ValueError: If search space has fewer than 2 dimensions
    """
    if len(search_space) < 2:
        raise ValueError("Search space must be at least two dimensional")

    # Separate multi-value and single-value dimensions
    search_space_2d = {}
    para_dict_set_values = {}

    for para_name, dim_values in search_space.items():
        if len(dim_values) == 1:
            para_dict_set_values[para_name] = dim_values[0]
        else:
            search_space_2d[para_name] = dim_values

    # Take first two multi-value dimensions
    if len(search_space_2d) < 2:
        raise ValueError("Need at least 2 dimensions with multiple values")

    para_names = list(search_space_2d.keys())[:2]
    search_space_2d = {name: search_space_2d[name] for name in para_names}

    para1, para2 = para_names
    (x_all, y_all) = search_space_2d.values()
    xi, yi = np.meshgrid(x_all, y_all)
    zi = []

    for dim_value1 in search_space_2d[para1]:
        zi_row = []
        for dim_value2 in search_space_2d[para2]:
            para_dict_2d = {para1: dim_value1, para2: dim_value2}
            para_dict = {**para_dict_2d, **para_dict_set_values}
            zi_row.append(objective_function(para_dict))
        zi.append(zi_row)

    zi = np.array(zi).T

    fig = go.Figure(
        data=go.Surface(
            z=zi,
            x=xi,
            y=yi,
            colorscale=color_scale,
        ),
    )

    if contour:
        fig.update_traces(
            contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=para1,
            yaxis_title=para2,
            zaxis_title="Metric",
        ),
        width=width,
        height=height,
    )
    return fig


def _plotly_surface(
    objective_function,
    search_space: Dict[str, np.ndarray],
    title: str = "Objective Function Surface",
    width: int = 900,
    height: int = 900,
    contour: bool = False,
) -> go.Figure:
    """Create a 3D surface plot for a 2D objective function.

    Args:
        objective_function: Function that takes a dict of parameters
        search_space: Dictionary with exactly 2 keys mapping to value arrays
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels
        contour: Whether to add contour projections

    Returns:
        Plotly Figure object

    Raises:
        ValueError: If search space is not exactly 2-dimensional
    """
    if len(search_space) != 2:
        raise ValueError("Search space must be exactly two dimensional")

    xi, yi, zi = _create_grid(objective_function, search_space)

    param_names = list(search_space.keys())

    fig = go.Figure(
        data=go.Surface(
            z=zi,
            x=xi,
            y=yi,
            colorscale=color_scale,
        ),
    )

    if contour:
        fig.update_traces(
            contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=param_names[0],
            yaxis_title=param_names[1],
            zaxis_title="Metric",
        ),
        width=width,
        height=height,
    )
    return fig


def _plotly_heatmap(
    objective_function,
    search_space: Dict[str, np.ndarray],
    title: str = "Objective Function Heatmap",
    width: int = 900,
    height: int = 900,
) -> go.Figure:
    """Create a 2D heatmap visualization of an objective function.

    Args:
        objective_function: Function that takes a dict of parameters
        search_space: Dictionary with exactly 2 keys mapping to value arrays
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        Plotly Figure object

    Raises:
        ValueError: If search space is not exactly 2-dimensional
    """
    if len(search_space) != 2:
        raise ValueError("Search space must be exactly two dimensional")

    xi, yi, zi = _create_grid(objective_function, search_space)
    param_names = list(search_space.keys())
    param_values = list(search_space.values())

    fig = px.imshow(
        img=zi,
        x=param_values[0],
        y=param_values[1],
        labels=dict(x=param_names[0], y=param_names[1], color="Metric"),
        color_continuous_scale=color_scale,
    )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
    )

    return fig


def _matplotlib_heatmap(
    objective_function,
    search_space: Dict[str, np.ndarray],
    title: str = "Objective Function Heatmap",
    norm: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Create a matplotlib heatmap visualization.

    Args:
        objective_function: Function that takes a dict of parameters
        search_space: Dictionary with exactly 2 keys mapping to value arrays
        title: Plot title
        norm: Normalization ('color_log' for LogNorm, None for linear)
        figsize: Figure size as (width, height)

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If search space is not exactly 2-dimensional
    """
    if len(search_space) != 2:
        raise ValueError("Search space must be exactly two dimensional")

    if norm == "color_log":
        norm = mpl.colors.LogNorm()

    xi, yi, zi = _create_grid(objective_function, search_space)
    param_names = list(search_space.keys())
    param_values = list(search_space.values())

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        zi,
        cmap=plt.cm.jet,
        extent=[
            param_values[0][0],
            param_values[0][-1],
            param_values[1][0],
            param_values[1][-1],
        ],
        aspect="auto",
        norm=norm,
        origin="lower",
    )

    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Metric")
    fig.tight_layout()
    return fig


def _matplotlib_surface(
    objective_function,
    search_space: Dict[str, np.ndarray],
    title: str = "Objective Function Surface",
    norm: Optional[str] = None,
    figsize: tuple = (10, 8),
    view_init: tuple = (30, 45),
) -> plt.Figure:
    """Create a matplotlib 3D surface plot.

    Args:
        objective_function: Function that takes a dict of parameters
        search_space: Dictionary with exactly 2 keys mapping to value arrays
        title: Plot title
        norm: Normalization ('color_log' for LogNorm, None for linear)
        figsize: Figure size as (width, height)
        view_init: 3D view angles as (elevation, azimuth)

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If search space is not exactly 2-dimensional
    """
    if len(search_space) != 2:
        raise ValueError("Search space must be exactly two dimensional")

    if norm == "color_log":
        norm = mpl.colors.LogNorm()

    xi, yi, zi = _create_grid(objective_function, search_space)
    param_names = list(search_space.keys())

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)

    surf = ax.plot_surface(
        xi,
        yi,
        zi,
        cmap=plt.cm.jet,
        cstride=1,
        rstride=1,
        antialiased=True,
        alpha=0.9,
        norm=norm,
    )

    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_zlabel("Metric")
    ax.set_title(title)
    ax.view_init(*view_init)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    fig.tight_layout()
    return fig


def _plotly_contour(
    objective_function,
    search_space: Dict[str, np.ndarray],
    title: str = "Objective Function Contours",
    width: int = 900,
    height: int = 900,
    contour_levels: int = 20,
) -> go.Figure:
    """Create a 2D contour plot visualization.

    Args:
        objective_function: Function that takes a dict of parameters
        search_space: Dictionary with exactly 2 keys mapping to value arrays
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels
        contour_levels: Number of contour lines

    Returns:
        Plotly Figure object

    Raises:
        ValueError: If search space is not exactly 2-dimensional
    """
    if len(search_space) != 2:
        raise ValueError("Search space must be exactly two dimensional")

    xi, yi, zi = _create_grid(objective_function, search_space)
    param_names = list(search_space.keys())

    fig = go.Figure(
        data=go.Contour(
            z=zi,
            x=search_space[param_names[0]],
            y=search_space[param_names[1]],
            colorscale=color_scale,
            contours=dict(
                start=np.min(zi),
                end=np.max(zi),
                size=(np.max(zi) - np.min(zi)) / contour_levels,
            ),
            line=dict(width=1),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=param_names[0],
        yaxis_title=param_names[1],
        width=width,
        height=height,
    )

    return fig


def _plot_parameter_slice(
    objective_function,
    search_space: Dict[str, np.ndarray],
    slice_param: str,
    fixed_params: Dict[str, float],
    title: str = "Parameter Slice",
    width: int = 800,
    height: int = 400,
) -> go.Figure:
    """Plot objective function values along one parameter dimension.

    Args:
        objective_function: Function that takes a dict of parameters
        search_space: Dictionary mapping parameter names to value arrays
        slice_param: Parameter name to vary along x-axis
        fixed_params: Fixed values for all other parameters
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    if slice_param not in search_space:
        raise ValueError(f"Parameter {slice_param} not found in search space")

    x_values = search_space[slice_param]
    y_values = []

    for x_val in x_values:
        params = fixed_params.copy()
        params[slice_param] = x_val
        y_values.append(objective_function(params))

    fig = go.Figure(
        data=go.Scatter(
            x=x_values, y=y_values, mode="lines+markers", line=dict(width=2), marker=dict(size=4)
        )
    )

    fig.update_layout(
        title=f"{title} - {slice_param}",
        xaxis_title=slice_param,
        yaxis_title="Metric",
        width=width,
        height=height,
    )

    return fig


def _create_function_comparison(
    functions: List[Any],
    search_space: Dict[str, np.ndarray],
    plot_type: str = "surface",
    title: str = "Function Comparison",
) -> List[go.Figure]:
    """Create comparison plots for multiple objective functions.

    Args:
        functions: List of objective function instances
        search_space: Common search space for all functions
        plot_type: Type of plot ('surface', 'heatmap', 'contour')
        title: Base title for plots

    Returns:
        List of Plotly Figure objects
    """
    figures = []

    for func in functions:
        func_title = f"{title} - {getattr(func, 'name', str(func))}"

        if plot_type == "surface":
            fig = _plotly_surface(func.objective_function, search_space, func_title)
        elif plot_type == "heatmap":
            fig = _plotly_heatmap(func.objective_function, search_space, func_title)
        elif plot_type == "contour":
            fig = _plotly_contour(func.objective_function, search_space, func_title)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        figures.append(fig)

    return figures


def _plotly_ml_hyperparameter_heatmap(
    ml_function,
    param1: str,
    param2: str,
    fixed_params: Dict[str, Any] = None,
    title: str = "ML Function Hyperparameter Analysis",
    width: int = 900,
    height: int = 700,
) -> go.Figure:
    """Create heatmap for ML function with 2 hyperparameters.

    Args:
        ml_function: ML test function instance
        param1: First hyperparameter name (x-axis)
        param2: Second hyperparameter name (y-axis)
        fixed_params: Fixed values for other parameters
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    search_space = ml_function.search_space()
    fixed_params = fixed_params or {}

    # Get parameter ranges
    param1_values = search_space[param1]
    param2_values = search_space[param2]

    # SINGLE VALIDATION TEST - fail fast if configuration is wrong
    test_params = fixed_params.copy()
    test_params[param1] = param1_values[0]
    test_params[param2] = param2_values[0]

    # Fill in missing required parameters
    for param_name in search_space:
        if param_name not in test_params:
            test_params[param_name] = search_space[param_name][0]

    # Test once - if this fails, the whole configuration is wrong
    ml_function.objective_function(test_params)

    # Create evaluation grid
    results = []

    for p1_val in param1_values:
        row_results = []

        for p2_val in param2_values:
            params = fixed_params.copy()
            params[param1] = p1_val
            params[param2] = p2_val

            # Fill in missing required parameters
            for param_name in search_space:
                if param_name not in params:
                    params[param_name] = search_space[param_name][0]

            score = ml_function.objective_function(params)
            row_results.append(float(score))

        results.append(row_results)

    # Convert to numpy and create plot
    z_values = np.array(results)

    # Handle categorical parameters for display
    if isinstance(param1_values[0], str):
        x_labels = param1_values
        x_values = list(range(len(param1_values)))
    else:
        x_labels = [str(v) for v in param1_values]
        x_values = param1_values

    if isinstance(param2_values[0], str):
        y_labels = param2_values
        y_values = list(range(len(param2_values)))
    else:
        y_labels = [str(v) for v in param2_values]
        y_values = param2_values

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_values,
            y=y_values,
            colorscale="Viridis",
            hoverongaps=False,
            hovertemplate=f"{param1}: %{{x}}<br>{param2}: %{{y}}<br>Score: %{{z:.4f}}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{title}<br>{param1} vs {param2}",
        xaxis_title=param1,
        yaxis_title=param2,
        width=width,
        height=height,
    )

    # Handle categorical axis labels
    if isinstance(param1_values[0], str):
        fig.update_xaxes(tickmode="array", tickvals=x_values, ticktext=x_labels)
    if isinstance(param2_values[0], str):
        fig.update_yaxes(tickmode="array", tickvals=y_values, ticktext=y_labels)

    return fig


def _plotly_dataset_hyperparameter_analysis(
    ml_function,
    hyperparameter: str,
    fixed_params: Dict[str, Any] = None,
    title: str = "Dataset vs Hyperparameter Analysis",
    width: int = 1000,
    height: int = 700,
) -> go.Figure:
    """Create visualization showing hyperparameter effect across datasets.

    Args:
        ml_function: ML test function instance
        hyperparameter: Hyperparameter name to analyze
        fixed_params: Fixed values for other parameters
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    search_space = ml_function.search_space()
    fixed_params = fixed_params or {}

    # SINGLE VALIDATION TEST - fail fast if configuration is wrong
    test_params = fixed_params.copy()
    test_params["dataset"] = search_space["dataset"][0]
    test_params[hyperparameter] = search_space[hyperparameter][0]

    # Fill in missing required parameters
    for param_name in search_space:
        if param_name not in test_params:
            test_params[param_name] = search_space[param_name][0]

    # Test once - if this fails, the whole configuration is wrong
    ml_function.objective_function(test_params)

    datasets = search_space["dataset"]
    hyperparameter_values = search_space[hyperparameter]

    # Get dataset names
    dataset_names = []
    for dataset_func in datasets:
        name = dataset_func.__name__.replace("_data", "").replace("_", " ").title()
        dataset_names.append(name)

    # Evaluate across datasets and hyperparameter values
    results = []

    for dataset_func in datasets:
        dataset_results = []

        for hyperparam_val in hyperparameter_values:
            params = fixed_params.copy()
            params["dataset"] = dataset_func
            params[hyperparameter] = hyperparam_val

            # Fill in missing required parameters with defaults
            for param_name in search_space:
                if param_name not in params:
                    params[param_name] = search_space[param_name][0]

            score = ml_function.objective_function(params)
            dataset_results.append(float(score))

        results.append(dataset_results)

    # Create heatmap
    z_values = np.array(results)

    # Handle categorical hyperparameter
    if isinstance(hyperparameter_values[0], str):
        x_labels = hyperparameter_values
        x_values = list(range(len(hyperparameter_values)))
    else:
        x_labels = [str(v) for v in hyperparameter_values]
        x_values = hyperparameter_values

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_values,
            y=list(range(len(dataset_names))),
            colorscale="Viridis",
            hoverongaps=False,
            hovertemplate=f"Dataset: %{{y}}<br>{hyperparameter}: %{{x}}<br>Score: %{{z:.4f}}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{title}<br>Impact of {hyperparameter} across Datasets",
        xaxis_title=hyperparameter,
        yaxis_title="Dataset",
        width=width,
        height=height,
    )

    # Set dataset names on y-axis
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(len(dataset_names))), ticktext=dataset_names
    )

    # Handle categorical x-axis
    if isinstance(hyperparameter_values[0], str):
        fig.update_xaxes(tickmode="array", tickvals=x_values, ticktext=x_labels)

    return fig


def _create_ml_function_analysis_suite(
    ml_function,
    output_dir: str = "ml_analysis_plots",
) -> Dict[str, go.Figure]:
    """Create comprehensive analysis suite for ML function.

    Args:
        ml_function: ML test function instance
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to Plotly figures
    """
    search_space = ml_function.search_space()
    figures = {}

    # Get hyperparameters (exclude dataset and cv)
    numeric_params = []
    categorical_params = []

    for param_name, param_values in search_space.items():
        if param_name in ["dataset", "cv"]:
            continue
        if isinstance(param_values[0], (int, float)):
            numeric_params.append(param_name)
        else:
            categorical_params.append(param_name)

    # 1. Hyperparameter vs Hyperparameter plots
    for i, param1 in enumerate(numeric_params + categorical_params):
        for j, param2 in enumerate(numeric_params + categorical_params):
            if i >= j:  # Avoid duplicates and self-comparisons
                continue

            plot_name = f"hyperparam_{param1}_vs_{param2}"
            fig = _plotly_ml_hyperparameter_heatmap(
                ml_function, param1, param2, title=f"{ml_function.name} - Hyperparameter Analysis"
            )
            figures[plot_name] = fig

    # 2. Dataset vs Hyperparameter plots
    for param_name in numeric_params + categorical_params:
        plot_name = f"dataset_vs_{param_name}"
        fig = _plotly_dataset_hyperparameter_analysis(
            ml_function, param_name, title=f"{ml_function.name} - Dataset Analysis"
        )
        figures[plot_name] = fig

    return figures
