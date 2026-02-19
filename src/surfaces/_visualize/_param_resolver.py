# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Parameter resolution for plot configuration.

This module handles the interpretation of the params dict passed to plot methods,
resolving defaults and validating dimension requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..test_functions._base_test_function import BaseTestFunction


@dataclass
class DimensionConfig:
    """Configuration for a single dimension."""

    name: str
    values: np.ndarray  # The values to use (for plotting or the single fixed value)
    is_plotted: bool  # True if this dimension is being plotted, False if fixed
    bounds: Tuple[float, float]  # (min, max) of the values

    @property
    def center(self) -> float:
        """Center value of the dimension (midpoint of bounds)."""
        return (self.bounds[0] + self.bounds[1]) / 2


@dataclass
class ResolvedParams:
    """Resolved parameter configuration for plotting.

    This dataclass contains all the information needed to generate a plot,
    with all defaults resolved and dimensions categorized.
    """

    plot_dims: List[DimensionConfig] = field(default_factory=list)
    fixed_dims: List[DimensionConfig] = field(default_factory=list)

    @property
    def all_dims(self) -> List[DimensionConfig]:
        """All dimensions in order (plotted first, then fixed)."""
        return self.plot_dims + self.fixed_dims

    @property
    def plot_dim_names(self) -> List[str]:
        """Names of dimensions being plotted."""
        return [d.name for d in self.plot_dims]

    @property
    def fixed_dim_names(self) -> List[str]:
        """Names of fixed dimensions."""
        return [d.name for d in self.fixed_dims]

    @property
    def fixed_values(self) -> Dict[str, float]:
        """Dictionary of fixed dimension names to their values."""
        return {d.name: d.values[0] for d in self.fixed_dims}

    def get_dim(self, name: str) -> Optional[DimensionConfig]:
        """Get dimension config by name."""
        for dim in self.all_dims:
            if dim.name == name:
                return dim
        return None


def resolve_params(
    func: "BaseTestFunction",
    params: Optional[Dict[str, Any]] = None,
    required_plot_dims: Optional[int] = None,
    resolution: int = 50,
    plot_all_by_default: bool = False,
) -> ResolvedParams:
    """Resolve params dict to a complete plot configuration.

    Parameters
    ----------
    func : BaseTestFunction
        The test function being plotted.
    params : dict, optional
        User-provided dimension configuration.
    required_plot_dims : int, optional
        If specified, validate that exactly this many dimensions are plotted.
    resolution : int, default=50
        Default number of points when generating ranges from tuples.
    plot_all_by_default : bool, default=False
        If True and no params specified, plot all dimensions instead of
        default_plot_dims. Useful for multi_slice which shows all dimensions.

    Returns
    -------
    ResolvedParams
        Complete configuration with all defaults resolved.

    Raises
    ------
    ValueError
        If required_plot_dims is specified and not satisfied.
    """
    # Get all dimension names from the function
    all_dim_names = _get_dimension_names(func)

    # Get defaults from the function
    default_plot_dims = _get_default_plot_dims(func, all_dim_names)
    default_bounds = _get_default_bounds(func, all_dim_names)
    default_fixed = _get_default_fixed(func, all_dim_names, default_bounds)
    default_step = _get_default_step(func, all_dim_names, default_bounds)

    # If no params provided, use complete defaults
    if params is None:
        params = {}

    # Determine which dimensions to plot
    plot_dim_names = _determine_plot_dims(
        params, all_dim_names, default_plot_dims, required_plot_dims, plot_all_by_default
    )

    # Build resolved configuration
    resolved = ResolvedParams()

    for dim_name in all_dim_names:
        is_plotted = dim_name in plot_dim_names
        user_value = params.get(dim_name)

        if is_plotted:
            # This dimension is being plotted
            values, bounds = _resolve_plot_dimension(
                dim_name,
                user_value,
                default_bounds.get(dim_name, (-5.0, 5.0)),
                default_step.get(dim_name),
                resolution,
                func,
            )
            resolved.plot_dims.append(
                DimensionConfig(
                    name=dim_name,
                    values=values,
                    is_plotted=True,
                    bounds=bounds,
                )
            )
        else:
            # This dimension is fixed
            fixed_value = _resolve_fixed_dimension(
                dim_name,
                user_value,
                default_fixed.get(dim_name, 0.0),
            )
            resolved.fixed_dims.append(
                DimensionConfig(
                    name=dim_name,
                    values=np.array([fixed_value]),
                    is_plotted=False,
                    bounds=(fixed_value, fixed_value),
                )
            )

    # Validate dimension count if required
    if required_plot_dims is not None:
        actual = len(resolved.plot_dims)
        if actual != required_plot_dims:
            raise ValueError(
                f"This plot requires exactly {required_plot_dims} plotted dimensions, "
                f"but {actual} were configured. "
                f"Plotted: {resolved.plot_dim_names}. "
                f"Use params={{'{all_dim_names[0]}': ..., '{all_dim_names[1]}': ...}} "
                f"to specify which dimensions to plot."
            )

    return resolved


def _get_dimension_names(func: "BaseTestFunction") -> List[str]:
    """Get ordered list of dimension names from function."""
    if hasattr(func, "search_space"):
        return list(func.search_space.keys())
    raise ValueError(f"Cannot determine dimensions for {type(func).__name__}")


def _get_default_plot_dims(func: "BaseTestFunction", all_dims: List[str]) -> List[str]:
    """Get default dimensions to plot."""
    if hasattr(func, "default_plot_dims"):
        return func.default_plot_dims

    # Default: first 2 dimensions (or all if less than 2)
    return all_dims[:2] if len(all_dims) >= 2 else all_dims


def _get_default_bounds(
    func: "BaseTestFunction", all_dims: List[str]
) -> Dict[str, Tuple[float, float]]:
    """Get default bounds for each dimension."""
    if hasattr(func, "default_bounds_per_dim"):
        return func.default_bounds_per_dim

    # Try to get from search_space
    result = {}
    if hasattr(func, "search_space"):
        for dim_name in all_dims:
            values = func.search_space.get(dim_name)
            if values is not None and len(values) > 0:
                # Handle both arrays and lists (including categorical)
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    result[dim_name] = (min(numeric_values), max(numeric_values))
                else:
                    # Categorical: use indices
                    result[dim_name] = (0, len(values) - 1)

    # Fall back to global default_bounds
    if hasattr(func, "default_bounds"):
        global_bounds = func.default_bounds
        for dim_name in all_dims:
            if dim_name not in result:
                result[dim_name] = global_bounds

    return result


def _get_default_fixed(
    func: "BaseTestFunction",
    all_dims: List[str],
    default_bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """Get default fixed values for each dimension."""
    if hasattr(func, "default_fixed"):
        return func.default_fixed

    # Default: center of bounds
    result = {}
    for dim_name in all_dims:
        if dim_name in default_bounds:
            min_val, max_val = default_bounds[dim_name]
            result[dim_name] = (min_val + max_val) / 2
        else:
            result[dim_name] = 0.0

    return result


def _get_default_step(
    func: "BaseTestFunction",
    all_dims: List[str],
    default_bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, Optional[float]]:
    """Get default step size for each dimension."""
    if hasattr(func, "default_step"):
        return func.default_step

    # Default: (max - min) / 100 for continuous
    result = {}
    for dim_name in all_dims:
        if dim_name in default_bounds:
            min_val, max_val = default_bounds[dim_name]
            result[dim_name] = (max_val - min_val) / 100
        else:
            result[dim_name] = None

    return result


def _determine_plot_dims(
    params: Dict[str, Any],
    all_dims: List[str],
    default_plot_dims: List[str],
    required_plot_dims: Optional[int],
    plot_all_by_default: bool = False,
) -> List[str]:
    """Determine which dimensions should be plotted based on params."""
    # Find dimensions explicitly marked for plotting
    explicit_plot_dims = []
    explicit_fixed_dims = []

    for dim_name, value in params.items():
        if dim_name not in all_dims:
            raise ValueError(f"Unknown dimension '{dim_name}'. " f"Available: {all_dims}")

        if _is_plot_marker(value):
            explicit_plot_dims.append(dim_name)
        else:
            explicit_fixed_dims.append(dim_name)

    # If explicit plot dims specified, use those
    if explicit_plot_dims:
        return explicit_plot_dims

    # If no explicit plot dims but some fixed dims, infer plot dims
    if explicit_fixed_dims and not explicit_plot_dims:
        # Plot all dims that aren't explicitly fixed
        remaining = [d for d in all_dims if d not in explicit_fixed_dims]
        if required_plot_dims is not None:
            return remaining[:required_plot_dims]
        return remaining

    # No params specified, use defaults
    # If plot_all_by_default is True, use all dims (for multi_slice)
    if plot_all_by_default:
        return all_dims

    if required_plot_dims is not None:
        return default_plot_dims[:required_plot_dims]
    return default_plot_dims


def _is_plot_marker(value: Any) -> bool:
    """Check if a value indicates the dimension should be plotted."""
    # Ellipsis means "plot with defaults"
    if value is ...:
        return True

    # Tuple of 2 numbers means range
    if isinstance(value, tuple) and len(value) == 2:
        return True

    # List or array means explicit values to plot
    if isinstance(value, (list, np.ndarray)) and len(value) > 1:
        return True

    # range() object
    if isinstance(value, range):
        return True

    # None means "use default" - interpret as plot if it's in default_plot_dims
    # This is handled elsewhere

    return False


def _resolve_plot_dimension(
    dim_name: str,
    user_value: Any,
    default_bounds: Tuple[float, float],
    default_step: Optional[float],
    resolution: int,
    func: "BaseTestFunction",
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Resolve a dimension that should be plotted to its values."""
    # Ellipsis or None: use defaults
    if user_value is None or user_value is ...:
        min_val, max_val = default_bounds
        values = np.linspace(min_val, max_val, resolution)
        return values, (min_val, max_val)

    # Tuple: range with default step
    if isinstance(user_value, tuple) and len(user_value) == 2:
        min_val, max_val = user_value
        values = np.linspace(min_val, max_val, resolution)
        return values, (min_val, max_val)

    # range(): explicit discrete values
    if isinstance(user_value, range):
        values = np.array(list(user_value))
        return values, (values.min(), values.max())

    # List or array: explicit values
    if isinstance(user_value, (list, np.ndarray)):
        values = np.array(user_value)
        # Filter to numeric values only (including numpy types)
        numeric_mask = [isinstance(v, (int, float, np.integer, np.floating)) for v in user_value]
        if not all(numeric_mask):
            # For categorical, we might need different handling
            # For now, use indices
            values = np.arange(len(user_value))
        return values, (float(values.min()), float(values.max()))

    raise ValueError(
        f"Invalid value for plotted dimension '{dim_name}': {user_value}. "
        f"Use a tuple (min, max), list of values, range(), or ... for defaults."
    )


def _resolve_fixed_dimension(
    dim_name: str,
    user_value: Any,
    default_fixed: float,
) -> float:
    """Resolve a dimension that should be fixed to its value."""
    # None or not specified: use default
    if user_value is None:
        return default_fixed

    # Single numeric value
    if isinstance(user_value, (int, float)):
        return float(user_value)

    # For categorical: could be any value
    # Return as-is and let the plotting handle it
    return user_value
