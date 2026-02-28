# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Compatibility system for matching plots with test functions.

This module decouples plot availability logic from the plot implementations
themselves. Instead of each plot method doing its own validation, we declare
requirements declaratively in PLOT_REGISTRY and check them uniformly.

Architecture
------------

    PlotAccessor.available()
        |
        v
    available_plots(func)          iterates PLOT_REGISTRY
        |
        v
    PlotRequirements.check(func)   checks dimensions, history, attributes
        |
        v
    (compatible, reason)           returned to caller

The PlotAccessor (in _accessor.py) exposes func.plot.available() which
delegates here. The same check is used by the documentation generator
(generate_compatibility.py) to produce the compatibility matrix.

Adding a new plot type
----------------------
1. Add a PlotRequirements entry to PLOT_REGISTRY
2. Implement the plot function in its own module
3. Add the corresponding method to PlotAccessor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..test_functions._base_test_function import BaseTestFunction


@dataclass
class PlotRequirements:
    """Declarative specification of when a plot type is available.

    Each plot type in PLOT_REGISTRY has one PlotRequirements instance that
    describes the conditions under which it can be used. The ``check()``
    method evaluates all conditions against a concrete function instance
    and returns (True, None) or (False, "human-readable reason").

    Parameters
    ----------
    name : str
        Identifier matching the PlotAccessor method name (e.g. "surface").
    description : str
        One-line description shown in func.plot.available() output and docs.
    dimensions : int or (min, max)
        Dimensionality constraint on the test function.

        - ``int``: function must have exactly this many dimensions.
          Example: ``dimensions=2`` for surface/contour/heatmap.
        - ``(min, max)`` tuple: function dimensions must fall in range.
          Use ``None`` for unbounded. Example: ``(1, None)`` means 1D+.
    requires_history : bool
        If True, the plot needs evaluation history (func.search_data or
        explicitly passed data). ``check()`` will reject when no history
        is available.
    requires_attribute : str or None
        If set, the function instance must have this attribute.
        Example: ``requires_attribute="latex_formula"`` ensures the plot
        is only available for functions that define a LaTeX formula.

    Examples
    --------
    Exact dimension match (2D only):

        >>> req = PlotRequirements("surface", "3D surface", dimensions=2)
        >>> req.check(SphereFunction(n_dim=2))
        (True, None)
        >>> req.check(SphereFunction(n_dim=5))
        (False, 'requires exactly 2D function, got 5D')

    Dimension range (1D and above):

        >>> req = PlotRequirements("multi_slice", "Slices", dimensions=(1, None))
        >>> req.check(ForresterFunction())  # 1D
        (True, None)

    Attribute gate:

        >>> req = PlotRequirements("latex", "PDF", dimensions=2,
        ...                        requires_attribute="latex_formula")
        >>> req.check(AckleyFunction())       # has latex_formula
        (True, None)
        >>> req.check(BBOBSphere(n_dim=2))    # no latex_formula
        (False, "requires 'latex_formula' attribute")
    """

    name: str
    description: str
    dimensions: Union[int, Tuple[Optional[int], Optional[int]]]
    requires_history: bool = False
    requires_attribute: Optional[str] = None

    def check(
        self, func: "BaseTestFunction", has_history: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate all requirements against a function instance.

        Checks are evaluated in order: dimensions, history, attribute.
        Returns on the first failing check.

        Parameters
        ----------
        func : BaseTestFunction
            The test function instance to check.
        has_history : bool
            Whether evaluation history data is available.

        Returns
        -------
        (bool, str or None)
            ``(True, None)`` if compatible, ``(False, reason)`` if not.
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

        # Check required attribute
        if self.requires_attribute and not hasattr(func, self.requires_attribute):
            return (
                False,
                f"requires '{self.requires_attribute}' attribute",
            )

        return (True, None)


# Central registry mapping plot name -> requirements.
#
# Keys must match the method names on PlotAccessor (in _accessor.py).
# To add a new plot type, add an entry here and a method on PlotAccessor.
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
        requires_attribute="latex_formula",
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

    # Try spec (SpecAccessor has .get() method)
    if hasattr(func, "spec"):
        n_dim = func.spec.get("n_dim")
        if n_dim is not None:
            return n_dim

    raise ValueError(f"Cannot determine dimensions for function: {type(func).__name__}")


def available_plots(func: "BaseTestFunction", has_history: bool = False) -> List[Dict[str, Any]]:
    """Get list of plots compatible with the given function.

    Args:
        func: The test function to check compatibility for.
        has_history: Whether optimization history data is available.

    Returns:
        List of dicts with 'name' and 'description' for each compatible plot.

    Examples:
        >>> from surfaces.test_functions import SphereFunction
        >>> from surfaces._visualize import available_plots
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
        >>> from surfaces._visualize import check_compatibility
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
        >>> from surfaces._visualize import plot_info
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
