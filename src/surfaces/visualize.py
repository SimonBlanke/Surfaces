# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any
from tqdm import tqdm

color_scale = px.colors.sequential.Jet


def _create_grid(objective_function, search_space: Dict[str, np.ndarray]):
    """Create a 2D grid for visualization from a search space and objective function.
    
    Args:
        objective_function: Function that takes a dict of parameters and returns a scalar
        search_space: Dictionary with exactly 2 keys, each mapping to numpy arrays
        
    Returns:
        tuple: (xi, yi, zi) meshgrid arrays for plotting
    """
    def objective_function_np(*args):
        para = {}
        for arg, key in zip(args, search_space.keys()):
            para[key] = arg
        return objective_function(para)

    (x_all, y_all) = search_space.values()
    xi, yi = np.meshgrid(x_all, y_all)
    zi = objective_function_np(xi, yi)

    return xi, yi, zi


def plotly_surface_nd(
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

    pbar_total = len(search_space_2d[para1]) * len(search_space_2d[para2])
    pbar = tqdm(total=pbar_total, desc="Generating surface")
    
    for dim_value1 in search_space_2d[para1]:
        zi_row = []
        for dim_value2 in search_space_2d[para2]:
            para_dict_2d = {para1: dim_value1, para2: dim_value2}
            para_dict = {**para_dict_2d, **para_dict_set_values}
            zi_row.append(objective_function(para_dict))
            pbar.update(1)
        zi.append(zi_row)

    zi = np.array(zi).T
    pbar.close()

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
            contours_z=dict(
                show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
            )
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


def plotly_surface(
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
            contours_z=dict(
                show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
            )
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


def plotly_heatmap(
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


def matplotlib_heatmap(
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
            param_values[0][0], param_values[0][-1],
            param_values[1][0], param_values[1][-1],
        ],
        aspect="auto",
        norm=norm,
        origin='lower'
    )
    
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Metric')
    fig.tight_layout()
    return fig


def matplotlib_surface(
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
        xi, yi, zi,
        cmap=plt.cm.jet,
        cstride=1, rstride=1,
        antialiased=True,
        alpha=0.9,
        norm=norm,
    )
    
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_zlabel('Metric')
    ax.set_title(title)
    ax.view_init(*view_init)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    fig.tight_layout()
    return fig


def plotly_contour(
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

    fig = go.Figure(data=go.Contour(
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
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=param_names[0],
        yaxis_title=param_names[1],
        width=width,
        height=height,
    )
    
    return fig


def plot_parameter_slice(
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
    
    fig = go.Figure(data=go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=f"{title} - {slice_param}",
        xaxis_title=slice_param,
        yaxis_title="Metric",
        width=width,
        height=height,
    )
    
    return fig


def create_function_comparison(
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
            fig = plotly_surface(func.objective_function, search_space, func_title)
        elif plot_type == "heatmap": 
            fig = plotly_heatmap(func.objective_function, search_space, func_title)
        elif plot_type == "contour":
            fig = plotly_contour(func.objective_function, search_space, func_title)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
        figures.append(fig)
    
    return figures
