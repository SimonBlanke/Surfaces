# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Visualization module for Surfaces test functions.

This module provides various plot types for visualizing objective function
landscapes and optimization progress. It includes a compatibility system
that helps users discover which plots work with their specific functions.

Quick Start
-----------
>>> from surfaces.test_functions import SphereFunction, AckleyFunction
>>> from surfaces.visualize import auto_plot, available_plots

# See what plots work with your function
>>> func = SphereFunction(n_dim=5)
>>> available_plots(func)
[{'name': 'multi_slice', 'description': '...'},
 {'name': 'fitness_distribution', 'description': '...'}]

# Auto-select the best visualization
>>> fig = auto_plot(func)
>>> fig.show()

# For 2D functions, surface plots are available
>>> func_2d = AckleyFunction()
>>> fig = plot_surface(func_2d)
>>> fig.show()

Plot Types
----------
- surface: 3D surface plot (2D functions only)
- contour: 2D contour plot (2D functions only)
- multi_slice: 1D slices through each dimension (any N-D)
- convergence: Best-so-far vs evaluations (requires history)
- fitness_distribution: Histogram of sampled values (any N-D)
- latex: Publication-quality LaTeX/PDF with formula (2D algebraic only)

Discovery Functions
-------------------
- available_plots(func): List plots compatible with a function
- check_compatibility(func, plot_name): Check if specific plot works
- plot_info(plot_name): Get info about a plot type
- list_all_plots(): List all available plot types
- auto_plot(func): Automatically select best visualization
"""

from ._auto import auto_plot
from ._compatibility import (
    available_plots,
    check_compatibility,
    list_all_plots,
    plot_info,
)
from ._contour import plot_contour
from ._convergence import plot_convergence
from ._distribution import plot_fitness_distribution
from ._errors import (
    MissingDataError,
    MissingDependencyError,
    PlotCompatibilityError,
    VisualizationError,
)
from ._latex import plot_latex
from ._slices import plot_multi_slice
from ._surface import plot_surface

__all__ = [
    # Discovery functions
    "available_plots",
    "check_compatibility",
    "plot_info",
    "list_all_plots",
    # Plot functions
    "plot_surface",
    "plot_contour",
    "plot_multi_slice",
    "plot_convergence",
    "plot_fitness_distribution",
    "plot_latex",
    "auto_plot",
    # Errors
    "VisualizationError",
    "PlotCompatibilityError",
    "MissingDataError",
    "MissingDependencyError",
]
