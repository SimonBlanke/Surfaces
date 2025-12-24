# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Compatibility system for matching plots with test functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..test_functions._base_test_function import BaseTestFunction


@dataclass
class PlotRequirements:
    """Requirements for a specific plot type."""

    name: str
    description: str
    dimensions: Union[int, Tuple[Optional[int], Optional[int]]]
    requires_history: bool = False
    function_types: Tuple[str, ...] = ("algebraic", "ml", "cec", "bbob", "engineering")

    def check(
        self, func: "BaseTestFunction", has_history: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Check if a function is compatible with this plot.

        Args:
            func: The test function to check.
            has_history: Whether optimization history data is available.

        Returns:
            Tuple of (is_compatible, reason_if_not).
        """
        # Check dimensions
        func_dims = _get_function_dimensions(func)

        if isinstance(self.dimensions, int):
            if func_dims != self.dimensions:
                return (
                    False,
                    f"requires exactly {self.dimensions}D function, got {func_dims}D",
                )
        elif isinstance(self.dimensions, tuple):
            min_dim, max_dim = self.dimensions
            if min_dim is not None and func_dims < min_dim:
                return (
                    False,
                    f"requires at least {min_dim}D function, got {func_dims}D",
                )
            if max_dim is not None and func_dims > max_dim:
                return (
                    False,
                    f"requires at most {max_dim}D function, got {func_dims}D",
                )

        # Check history requirement
        if self.requires_history and not has_history:
            return (False, "requires optimization history data")

        return (True, None)


# Registry of all available plots and their requirements
PLOT_REGISTRY: Dict[str, PlotRequirements] = {
    "surface": PlotRequirements(
        name="surface",
        description="3D surface plot showing the objective landscape",
        dimensions=2,
    ),
    "contour": PlotRequirements(
        name="contour",
        description="2D contour plot with isolines of equal objective value",
        dimensions=2,
    ),
    "heatmap": PlotRequirements(
        name="heatmap",
        description="2D heatmap showing objective values as colors",
        dimensions=2,
    ),
    "multi_slice": PlotRequirements(
        name="multi_slice",
        description="Multiple 1D slices through each dimension",
        dimensions=(1, None),  # Works with 1D and up
    ),
    "convergence": PlotRequirements(
        name="convergence",
        description="Best-so-far objective value vs evaluation number",
        dimensions=(1, None),
        requires_history=True,
    ),
    "fitness_distribution": PlotRequirements(
        name="fitness_distribution",
        description="Histogram of objective values from random sampling",
        dimensions=(1, None),
    ),
    "latex": PlotRequirements(
        name="latex",
        description="Publication-quality LaTeX/PDF with pgfplots 3D surface and formula",
        dimensions=2,
        function_types=("algebraic",),  # Only algebraic functions have latex_formula
    ),
}


def _get_function_dimensions(func: "BaseTestFunction") -> int:
    """Extract the number of dimensions from a test function."""
    # Try n_dim attribute first
    if hasattr(func, "n_dim"):
        return func.n_dim

    # Try to infer from search space
    if hasattr(func, "search_space"):
        try:
            return len(func.search_space)
        except (TypeError, AttributeError):
            pass

    # Try spec
    if hasattr(func, "spec"):
        spec = func.spec
        if "n_dim" in spec and spec["n_dim"] is not None:
            return spec["n_dim"]

    raise ValueError(f"Cannot determine dimensions for function: {type(func).__name__}")


def _get_function_type(func: "BaseTestFunction") -> str:
    """Determine the type category of a test function."""
    module_name = type(func).__module__

    if "algebraic" in module_name:
        return "algebraic"
    elif "machine_learning" in module_name or "ml" in module_name:
        return "ml"
    elif "cec" in module_name:
        return "cec"
    elif "bbob" in module_name:
        return "bbob"
    elif "engineering" in module_name:
        return "engineering"
    else:
        return "unknown"


def available_plots(func: "BaseTestFunction", has_history: bool = False) -> List[Dict[str, Any]]:
    """Get list of plots compatible with the given function.

    Args:
        func: The test function to check compatibility for.
        has_history: Whether optimization history data is available.

    Returns:
        List of dicts with 'name' and 'description' for each compatible plot.

    Examples:
        >>> from surfaces.test_functions import SphereFunction
        >>> from surfaces.visualize import available_plots
        >>> func = SphereFunction(n_dim=2)
        >>> plots = available_plots(func)
        >>> [p['name'] for p in plots]
        ['surface', 'contour', 'heatmap', 'multi_slice', 'fitness_distribution']
    """
    compatible = []

    for name, requirements in PLOT_REGISTRY.items():
        is_compatible, _ = requirements.check(func, has_history)
        if is_compatible:
            compatible.append(
                {
                    "name": name,
                    "description": requirements.description,
                }
            )

    return compatible


def check_compatibility(
    func: "BaseTestFunction", plot_name: str, has_history: bool = False
) -> Tuple[bool, Optional[str]]:
    """Check if a specific plot is compatible with a function.

    Args:
        func: The test function to check.
        plot_name: Name of the plot type.
        has_history: Whether optimization history data is available.

    Returns:
        Tuple of (is_compatible, reason_if_not).

    Examples:
        >>> from surfaces.test_functions import SphereFunction
        >>> from surfaces.visualize import check_compatibility
        >>> func = SphereFunction(n_dim=5)
        >>> compatible, reason = check_compatibility(func, 'surface')
        >>> compatible
        False
        >>> reason
        'requires exactly 2D function, got 5D'
    """
    if plot_name not in PLOT_REGISTRY:
        return (False, f"unknown plot type: '{plot_name}'")

    return PLOT_REGISTRY[plot_name].check(func, has_history)


def plot_info(plot_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific plot type.

    Args:
        plot_name: Name of the plot type.

    Returns:
        Dict with plot information, or None if plot not found.

    Examples:
        >>> from surfaces.visualize import plot_info
        >>> info = plot_info('surface')
        >>> info['description']
        '3D surface plot showing the objective landscape'
    """
    if plot_name not in PLOT_REGISTRY:
        return None

    req = PLOT_REGISTRY[plot_name]
    return {
        "name": req.name,
        "description": req.description,
        "dimensions": req.dimensions,
        "requires_history": req.requires_history,
    }


def list_all_plots() -> List[Dict[str, Any]]:
    """Get information about all available plot types.

    Returns:
        List of dicts with information about each plot type.
    """
    return [plot_info(name) for name in PLOT_REGISTRY.keys()]
