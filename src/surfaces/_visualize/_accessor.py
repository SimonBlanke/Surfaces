# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Plot accessor for test functions.

This module provides the PlotAccessor class that enables the fluent
`func.plot.surface()` API pattern for test function visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..test_functions._base_test_function import BaseTestFunction


class PlotAccessor:
    """Namespace for plot methods on test functions.

    Provides IDE-discoverable plotting methods with runtime validation
    based on the function's dimensions and characteristics.

    This class implements the Accessor Pattern (similar to pandas DataFrame.plot)
    to provide a clean namespace for visualization methods without polluting
    the test function's interface.

    Parameters
    ----------
    func : BaseTestFunction
        The test function to visualize.

    Examples
    --------
    >>> from surfaces.test_functions import SphereFunction, AckleyFunction

    # Basic usage - all defaults
    >>> func = AckleyFunction()  # 2D function
    >>> fig = func.plot.surface()
    >>> fig = func.plot.contour()

    # N-dimensional function with explicit dimensions
    >>> func = SphereFunction(n_dim=5)
    >>> fig = func.plot.surface(params={"x0": ..., "x2": ...})

    # Custom ranges and fixed values
    >>> fig = func.plot.surface(params={
    ...     "x0": (-2, 2),
    ...     "x2": (-1, 1),
    ...     "x1": 0.0,
    ...     "x3": 0.5,
    ...     "x4": -1.0
    ... })
    """

    def __init__(self, func: "BaseTestFunction") -> None:
        self._func = func
        self._history: Optional[List[Dict[str, Any]]] = None

    def with_history(self, history: List[Dict[str, Any]]) -> "PlotAccessor":
        """Attach optimization history for convergence plots.

        Parameters
        ----------
        history : list of dict
            List of evaluation records, each containing parameters and 'score'.

        Returns
        -------
        PlotAccessor
            Self, for method chaining.

        Examples
        --------
        >>> func = SphereFunction(n_dim=2)
        >>> # ... run optimization, collect history ...
        >>> fig = func.plot.with_history(history).convergence()
        """
        self._history = history
        return self

    # =========================================================================
    # 2D Plots (require exactly 2 plotted dimensions)
    # =========================================================================

    def surface(
        self,
        params: Optional[Dict[str, Any]] = None,
        resolution: int = 50,
        **kwargs: Any,
    ):
        """3D surface plot of the objective landscape.

        Creates an interactive 3D surface visualization showing how the
        objective function value changes across two dimensions.

        Parameters
        ----------
        params : dict, optional
            Dimension configuration. For each dimension:
            - Tuple (min, max): Plot this dimension with given range
            - List/array: Plot this dimension with these values
            - range(): Plot with explicit discrete values
            - Single value: Fix this dimension at this value
            - Ellipsis (...): Plot with all defaults
            - Not specified: Fixed at default value

            Exactly 2 dimensions must have range/list/ellipsis values.

        resolution : int, default=50
            Number of points per axis when using tuple ranges.

        **kwargs
            Additional arguments passed to the underlying plot function
            (e.g., colorscale, title).

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive 3D surface plot.

        Raises
        ------
        ValueError
            If not exactly 2 dimensions are configured for plotting.

        Examples
        --------
        >>> func = AckleyFunction()  # 2D
        >>> fig = func.plot.surface()

        >>> func = SphereFunction(n_dim=5)
        >>> fig = func.plot.surface(params={"x0": ..., "x2": ...})
        >>> fig = func.plot.surface(params={
        ...     "x0": (-2, 2),
        ...     "x2": (-1, 1),
        ...     "x1": 0.0
        ... })
        """
        from ._param_resolver import resolve_params
        from ._surface import plot_surface

        resolved = resolve_params(self._func, params, required_plot_dims=2, resolution=resolution)
        return plot_surface(
            self._func,
            resolved=resolved,
            resolution=resolution,
            **kwargs,
        )

    def contour(
        self,
        params: Optional[Dict[str, Any]] = None,
        resolution: int = 50,
        **kwargs: Any,
    ):
        """2D contour plot with isolines of equal objective value.

        Creates a 2D visualization with contour lines showing regions
        of equal objective function value.

        Parameters
        ----------
        params : dict, optional
            Dimension configuration. See surface() for details.
            Exactly 2 dimensions must be configured for plotting.

        resolution : int, default=50
            Number of points per axis when using tuple ranges.

        **kwargs
            Additional arguments passed to the underlying plot function.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive contour plot.

        Examples
        --------
        >>> func = AckleyFunction()
        >>> fig = func.plot.contour()
        >>> fig = func.plot.contour(params={"x0": (-2, 2), "x1": (-2, 2)})
        """
        from ._contour import plot_contour
        from ._param_resolver import resolve_params

        resolved = resolve_params(self._func, params, required_plot_dims=2, resolution=resolution)
        return plot_contour(
            self._func,
            resolved=resolved,
            resolution=resolution,
            **kwargs,
        )

    def heatmap(
        self,
        params: Optional[Dict[str, Any]] = None,
        resolution: int = 50,
        **kwargs: Any,
    ):
        """2D heatmap showing objective values as colors.

        Creates a color-coded grid visualization of objective function values.

        Parameters
        ----------
        params : dict, optional
            Dimension configuration. See surface() for details.
            Exactly 2 dimensions must be configured for plotting.

        resolution : int, default=50
            Number of points per axis when using tuple ranges.

        **kwargs
            Additional arguments passed to the underlying plot function.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive heatmap plot.

        Examples
        --------
        >>> func = AckleyFunction()
        >>> fig = func.plot.heatmap()
        """
        from ._contour import plot_heatmap
        from ._param_resolver import resolve_params

        resolved = resolve_params(self._func, params, required_plot_dims=2, resolution=resolution)
        return plot_heatmap(
            self._func,
            resolved=resolved,
            resolution=resolution,
            **kwargs,
        )

    # =========================================================================
    # N-D Plots (work with any number of dimensions)
    # =========================================================================

    def multi_slice(
        self,
        params: Optional[Dict[str, Any]] = None,
        resolution: int = 50,
        **kwargs: Any,
    ):
        """1D slices through each dimension.

        Creates multiple 1D plots, one for each dimension, showing how
        the objective changes along that dimension while others are fixed.

        Parameters
        ----------
        params : dict, optional
            Dimension configuration. For slice plots:
            - Tuple (min, max): Range for this dimension's slice
            - Single value: Fix this dimension (not shown in plots)
            - Ellipsis (...): Use defaults for this dimension

            All dimensions are sliced by default. To exclude a dimension
            from the plots, fix it with a single value.

        resolution : int, default=50
            Number of points per slice.

        **kwargs
            Additional arguments passed to the underlying plot function.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with subplots for each dimension.

        Examples
        --------
        >>> func = SphereFunction(n_dim=5)
        >>> fig = func.plot.multi_slice()  # Shows all 5 dimensions
        >>> fig = func.plot.multi_slice(params={"x0": (-2, 2)})  # Custom range for x0
        >>> fig = func.plot.multi_slice(params={"x2": 0.0})  # Fix x2, show only 4 dims
        """
        from ._param_resolver import resolve_params
        from ._slices import plot_multi_slice

        # Use plot_all_by_default=True so all dimensions are shown by default
        resolved = resolve_params(
            self._func,
            params,
            required_plot_dims=None,
            resolution=resolution,
            plot_all_by_default=True,
        )
        return plot_multi_slice(
            self._func,
            resolved=resolved,
            resolution=resolution,
            **kwargs,
        )

    def fitness_distribution(
        self,
        params: Optional[Dict[str, Any]] = None,
        n_samples: int = 1000,
        **kwargs: Any,
    ):
        """Histogram of objective values from random sampling.

        Samples random points from the search space and shows the
        distribution of objective function values.

        Parameters
        ----------
        params : dict, optional
            Dimension configuration to restrict sampling region.

        n_samples : int, default=1000
            Number of random samples to draw.

        **kwargs
            Additional arguments passed to the underlying plot function.

        Returns
        -------
        plotly.graph_objects.Figure
            Histogram of sampled objective values.

        Examples
        --------
        >>> func = SphereFunction(n_dim=5)
        >>> fig = func.plot.fitness_distribution()
        """
        from ._distribution import plot_fitness_distribution
        from ._param_resolver import resolve_params

        resolved = resolve_params(self._func, params, required_plot_dims=None)
        return plot_fitness_distribution(
            self._func,
            resolved=resolved,
            n_samples=n_samples,
            **kwargs,
        )

    # =========================================================================
    # History-dependent Plots
    # =========================================================================

    def convergence(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        """Best-so-far objective value vs evaluation number.

        Shows optimization progress over time by plotting the best
        objective value found at each evaluation.

        Parameters
        ----------
        history : list of dict, optional
            Evaluation history. If not provided, uses history attached
            via with_history() or the function's search_data.

        **kwargs
            Additional arguments passed to the underlying plot function.

        Returns
        -------
        plotly.graph_objects.Figure
            Line plot of convergence.

        Raises
        ------
        ValueError
            If no history data is available.

        Examples
        --------
        >>> func = SphereFunction(n_dim=2)
        >>> # After optimization:
        >>> fig = func.plot.convergence()  # Uses func.search_data

        >>> # Or with explicit history:
        >>> fig = func.plot.convergence(history=my_history)

        >>> # Or via chaining:
        >>> fig = func.plot.with_history(my_history).convergence()
        """
        from ._convergence import plot_convergence

        # Priority: explicit argument > with_history() > search_data
        if history is None:
            history = self._history
        if history is None and hasattr(self._func, "search_data"):
            history = self._func.search_data

        if history is None or (hasattr(history, "__len__") and len(history) == 0):
            raise ValueError(
                "No history data available. Either:\n"
                "  1. Run some evaluations first (func.search_data)\n"
                "  2. Pass history explicitly: func.plot.convergence(history=...)\n"
                "  3. Use chaining: func.plot.with_history(history).convergence()"
            )

        return plot_convergence(self._func, history=history, **kwargs)

    # =========================================================================
    # Special Plots
    # =========================================================================

    def latex(
        self,
        params: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """Publication-quality LaTeX/PDF with pgfplots surface and formula.

        Generates a LaTeX document with a 3D surface plot using pgfplots
        and the function's mathematical formula.

        Parameters
        ----------
        params : dict, optional
            Dimension configuration. See surface() for details.
            Exactly 2 dimensions must be configured for plotting.

        output_path : str, optional
            Path for the output PDF file.

        **kwargs
            Additional arguments passed to the underlying plot function.

        Returns
        -------
        str
            Path to the generated PDF file.

        Raises
        ------
        ValueError
            If the function doesn't have a latex_formula attribute.
            If not exactly 2 dimensions are configured for plotting.

        Examples
        --------
        >>> func = AckleyFunction()
        >>> pdf_path = func.plot.latex()
        >>> pdf_path = func.plot.latex(output_path="my_plot.pdf")
        """
        from ._latex import plot_latex
        from ._param_resolver import resolve_params

        # Check for latex_formula
        if not hasattr(self._func, "latex_formula"):
            raise ValueError(
                f"{type(self._func).__name__} does not have a latex_formula attribute. "
                "The latex plot is only available for algebraic functions with formulas."
            )

        resolved = resolve_params(self._func, params, required_plot_dims=2)
        return plot_latex(
            self._func,
            resolved=resolved,
            output_path=output_path,
            **kwargs,
        )

    # =========================================================================
    # Discovery Methods
    # =========================================================================

    def available(self) -> List[str]:
        """List plot types available for this function.

        Returns
        -------
        list of str
            Names of plots compatible with this function.

        Examples
        --------
        >>> func = SphereFunction(n_dim=2)
        >>> func.plot.available()
        ['surface', 'contour', 'heatmap', 'multi_slice', 'fitness_distribution', 'latex']

        >>> func = SphereFunction(n_dim=5)
        >>> func.plot.available()
        ['multi_slice', 'fitness_distribution']
        """
        from ._compatibility import available_plots

        has_history = bool(self._history) or bool(getattr(self._func, "search_data", None))
        plots = available_plots(self._func, has_history=has_history)
        return [p["name"] for p in plots]
