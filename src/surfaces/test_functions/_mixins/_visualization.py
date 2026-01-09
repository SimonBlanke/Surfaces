# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Visualization mixin for test functions."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class VisualizationMixin:
    """Mixin providing visualization defaults and plot access for test functions.

    Provides the `.plot` accessor property and default values for visualization
    parameters like bounds, fixed values, and step sizes.

    Notes
    -----
    This mixin expects the following attribute to be present:
    - self.search_space: dict
    """

    @property
    def plot(self):
        """Access plotting methods for this function.

        Returns a PlotAccessor that provides plot methods filtered
        by this function's dimensions and characteristics.

        The accessor provides a fluent API for creating visualizations
        with sensible defaults and progressive customization.

        Returns
        -------
        PlotAccessor
            Accessor object with plotting methods.

        Examples
        --------
        Basic usage with defaults:

        >>> func = AckleyFunction()
        >>> fig = func.plot.surface()
        >>> fig = func.plot.contour()

        N-dimensional function with custom dimensions:

        >>> func = SphereFunction(n_dim=5)
        >>> fig = func.plot.surface(params={"x0": ..., "x2": ...})

        Custom ranges and fixed values:

        >>> fig = func.plot.surface(params={
        ...     "x0": (-2, 2),
        ...     "x2": (-1, 1),
        ...     "x1": 0.0,
        ... })

        Available plots for this function:

        >>> func.plot.available()
        ['surface', 'contour', 'heatmap', 'multi_slice', ...]
        """
        from surfaces.visualize._accessor import PlotAccessor

        return PlotAccessor(self)

    @property
    def dimensions(self) -> list:
        """List of dimension names for this function.

        Returns
        -------
        list of str
            Ordered list of dimension names (e.g., ['x0', 'x1'] or
            ['n_estimators', 'max_depth']).

        Examples
        --------
        >>> func = SphereFunction(n_dim=3)
        >>> func.dimensions
        ['x0', 'x1', 'x2']

        >>> func = RandomForestRegressorFunction()
        >>> func.dimensions
        ['n_estimators', 'max_depth', 'min_samples_split']
        """
        return list(self.search_space.keys())

    # =========================================================================
    # Visualization Defaults (can be overridden in subclasses)
    # =========================================================================

    @property
    def default_plot_dims(self) -> List[str]:
        """Default dimensions to plot (first 2 dimensions).

        Returns
        -------
        list of str
            Names of dimensions to plot by default.

        Examples
        --------
        >>> func = SphereFunction(n_dim=5)
        >>> func.default_plot_dims
        ['x0', 'x1']
        """
        dims = self.dimensions
        return dims[:2] if len(dims) >= 2 else dims

    @property
    def default_bounds_per_dim(self) -> Dict[str, Tuple[float, float]]:
        """Default bounds for each dimension.

        Bounds are inferred from the search space values:
        - For numeric values: (min, max) of the values
        - For categorical: (0, len-1) as index range

        Returns
        -------
        dict
            Mapping of dimension name to (min, max) tuple.

        Examples
        --------
        >>> func = SphereFunction(n_dim=2)
        >>> func.default_bounds_per_dim
        {'x0': (-5.0, 5.0), 'x1': (-5.0, 5.0)}
        """
        result = {}
        for dim_name, values in self.search_space.items():
            result[dim_name] = self._infer_bounds_for_dimension(values)
        return result

    @property
    def default_fixed(self) -> Dict[str, Any]:
        """Default values for fixed dimensions.

        Fixed values are inferred from the search space:
        - For continuous: center of bounds
        - For discrete: middle value from list

        Returns
        -------
        dict
            Mapping of dimension name to default fixed value.

        Examples
        --------
        >>> func = SphereFunction(n_dim=2)
        >>> func.default_fixed
        {'x0': 0.0, 'x1': 0.0}
        """
        result = {}
        for dim_name, values in self.search_space.items():
            result[dim_name] = self._infer_fixed_for_dimension(values)
        return result

    @property
    def default_step(self) -> Dict[str, Optional[float]]:
        """Default step size for each dimension.

        Step sizes are inferred based on dimension type:
        - Continuous (dense array): (max - min) / 100
        - Discrete numeric: minimum difference between consecutive values
        - Categorical: None (no step concept)

        Returns
        -------
        dict
            Mapping of dimension name to step size (or None for categorical).

        Examples
        --------
        >>> func = SphereFunction(n_dim=2)
        >>> func.default_step
        {'x0': 0.1, 'x1': 0.1}
        """
        result = {}
        for dim_name, values in self.search_space.items():
            result[dim_name] = self._infer_step_for_dimension(values)
        return result

    def _infer_bounds_for_dimension(self, values: Any) -> Tuple[float, float]:
        """Infer bounds for a single dimension based on its values."""
        if values is None or (hasattr(values, "__len__") and len(values) == 0):
            return (0.0, 1.0)

        # Filter to numeric values (including numpy types)
        numeric_values = [v for v in values if isinstance(v, (int, float, np.integer, np.floating))]

        if numeric_values:
            return (float(min(numeric_values)), float(max(numeric_values)))
        else:
            # Categorical: use index range
            return (0.0, float(len(values) - 1))

    def _infer_fixed_for_dimension(self, values: Any) -> Any:
        """Infer default fixed value for a single dimension."""
        if values is None or (hasattr(values, "__len__") and len(values) == 0):
            return 0.0

        # For arrays/lists, use middle value
        if hasattr(values, "__len__"):
            mid_idx = len(values) // 2
            return values[mid_idx]

        # Fallback
        return 0.0

    def _infer_step_for_dimension(self, values: Any) -> Optional[float]:
        """Infer step size for a single dimension based on its type.

        - Continuous (dense numpy array with >50 values): (max-min)/100
        - Discrete numeric: minimum difference between sorted values
        - Categorical: None
        """
        if values is None or (hasattr(values, "__len__") and len(values) < 2):
            return None

        # Check if continuous (dense numpy array)
        if isinstance(values, np.ndarray) and len(values) > 50:
            return float((values.max() - values.min()) / 100)

        # Filter to numeric values
        numeric_values = [v for v in values if isinstance(v, (int, float, np.integer, np.floating))]

        if len(numeric_values) >= 2:
            # Discrete numeric: infer step from consecutive differences
            sorted_vals = sorted(numeric_values)
            steps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
            return float(min(steps)) if steps else None

        # Categorical: no step concept
        return None
