# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Plot namespace for CustomTestFunction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .._custom_test_function import CustomTestFunction


def _check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "Visualization requires matplotlib. " "Install with: pip install matplotlib"
        )


class PlotNamespace:
    """Visualization tools for optimization results.

    This namespace provides methods for visualizing the optimization
    history, parameter importance, and optimization landscape.

    Parameters
    ----------
    func : CustomTestFunction
        The parent function to visualize.

    Examples
    --------
    >>> func.plot.history()
    >>> func.plot.contour("x", "y")
    >>> func.plot.importance()
    """

    def __init__(self, func: "CustomTestFunction") -> None:
        self._func = func

    def _check_data(self, min_evaluations: int = 1) -> None:
        """Check that sufficient data is available."""
        if self._func.n_evaluations < min_evaluations:
            raise ValueError(
                f"Plotting requires at least {min_evaluations} evaluations, "
                f"got {self._func.n_evaluations}"
            )

    def history(
        self,
        show_best: bool = True,
        log_scale: bool = False,
        figsize: Tuple[int, int] = (10, 6),
        ax=None,
    ):
        """Plot optimization history.

        Shows score values over evaluation number, optionally with
        running best overlay.

        Parameters
        ----------
        show_best : bool, default=True
            Show running best as overlay line.
        log_scale : bool, default=False
            Use logarithmic scale for y-axis.
        figsize : tuple, default=(10, 6)
            Figure size in inches.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.

        Examples
        --------
        >>> func.plot.history()
        >>> func.plot.history(log_scale=True, show_best=True)
        """
        plt = _check_matplotlib()
        self._check_data()

        X, y = self._func.get_data_as_arrays()
        evals = np.arange(1, len(y) + 1)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Plot all evaluations
        ax.scatter(evals, y, alpha=0.5, s=20, label="Evaluations")

        # Plot running best
        if show_best:
            if self._func.objective == "minimize":
                running_best = np.minimum.accumulate(y)
            else:
                running_best = np.maximum.accumulate(y)
            ax.plot(evals, running_best, "r-", linewidth=2, label="Best so far")

        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Score")
        ax.set_title(f"Optimization History ({self._func.n_evaluations} evaluations)")

        if log_scale:
            ax.set_yscale("log")

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return ax

    def importance(
        self,
        method: str = "variance",
        figsize: Tuple[int, int] = (8, 5),
        ax=None,
    ):
        """Plot parameter importance as bar chart.

        Parameters
        ----------
        method : str, default="variance"
            Importance calculation method.
        figsize : tuple, default=(8, 5)
            Figure size in inches.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.

        Examples
        --------
        >>> func.plot.importance()
        """
        plt = _check_matplotlib()
        self._check_data(min_evaluations=10)

        importance = self._func.analysis.parameter_importance(method=method)

        # Sort by importance
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        bars = ax.barh(names, values)
        ax.set_xlabel("Importance")
        ax.set_title("Parameter Importance")

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}",
                va="center",
            )

        ax.set_xlim(0, max(values) * 1.15)
        plt.tight_layout()
        return ax

    def contour(
        self,
        param_x: str,
        param_y: str,
        resolution: int = 50,
        show_points: bool = True,
        show_best: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        ax=None,
    ):
        """Plot 2D contour/heatmap of the optimization landscape.

        Uses interpolation of collected data points to create a
        continuous surface visualization.

        Parameters
        ----------
        param_x : str
            Parameter name for x-axis.
        param_y : str
            Parameter name for y-axis.
        resolution : int, default=50
            Grid resolution for interpolation.
        show_points : bool, default=True
            Show evaluation points as scatter.
        show_best : bool, default=True
            Highlight the best point.
        figsize : tuple, default=(10, 8)
            Figure size in inches.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.

        Examples
        --------
        >>> func.plot.contour("learning_rate", "n_estimators")
        """
        plt = _check_matplotlib()
        self._check_data(min_evaluations=10)

        # Validate parameters
        if param_x not in self._func.param_names:
            raise ValueError(f"Unknown parameter: {param_x}")
        if param_y not in self._func.param_names:
            raise ValueError(f"Unknown parameter: {param_y}")

        # Get data
        X, y = self._func.get_data_as_arrays()
        param_names = self._func.param_names
        idx_x = param_names.index(param_x)
        idx_y = param_names.index(param_y)

        x_data = X[:, idx_x]
        y_data = X[:, idx_y]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Create interpolated surface
        try:
            from scipy.interpolate import griddata

            # Create grid
            x_bounds = self._func.bounds[param_x]
            y_bounds = self._func.bounds[param_y]
            xi = np.linspace(x_bounds[0], x_bounds[1], resolution)
            yi = np.linspace(y_bounds[0], y_bounds[1], resolution)
            Xi, Yi = np.meshgrid(xi, yi)

            # Interpolate
            Zi = griddata((x_data, y_data), y, (Xi, Yi), method="cubic")

            # Plot contour
            contour = ax.contourf(Xi, Yi, Zi, levels=20, cmap="viridis")
            plt.colorbar(contour, ax=ax, label="Score")

        except ImportError:
            # Fallback: just scatter plot
            scatter = ax.scatter(x_data, y_data, c=y, cmap="viridis", s=50)
            plt.colorbar(scatter, ax=ax, label="Score")

        # Show evaluation points
        if show_points:
            ax.scatter(x_data, y_data, c="white", s=20, alpha=0.5, edgecolors="black")

        # Highlight best
        if show_best and self._func.best_params:
            best_x = self._func.best_params[param_x]
            best_y = self._func.best_params[param_y]
            ax.scatter(
                [best_x],
                [best_y],
                c="red",
                s=200,
                marker="*",
                edgecolors="white",
                linewidths=2,
                label=f"Best: {self._func.best_score:.4g}",
            )
            ax.legend()

        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_title(f"Optimization Landscape: {param_x} vs {param_y}")

        plt.tight_layout()
        return ax

    def parallel_coordinates(
        self,
        top_k: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 6),
        ax=None,
    ):
        """Plot parallel coordinates for high-dimensional visualization.

        Parameters
        ----------
        top_k : int, optional
            Only show top k evaluations. Shows all if None.
        figsize : tuple, default=(12, 6)
            Figure size in inches.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.

        Examples
        --------
        >>> func.plot.parallel_coordinates(top_k=50)
        """
        plt = _check_matplotlib()
        self._check_data(min_evaluations=5)

        X, y = self._func.get_data_as_arrays()
        param_names = self._func.param_names

        # Select top k if specified
        if top_k is not None:
            if self._func.objective == "minimize":
                indices = np.argsort(y)[:top_k]
            else:
                indices = np.argsort(y)[-top_k:]
            X = X[indices]
            y = y[indices]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Normalize each dimension to [0, 1]
        X_norm = np.zeros_like(X)
        for i in range(X.shape[1]):
            col = X[:, i]
            min_val, max_val = col.min(), col.max()
            if max_val > min_val:
                X_norm[:, i] = (col - min_val) / (max_val - min_val)
            else:
                X_norm[:, i] = 0.5

        # Color by score
        colors = plt.cm.viridis((y - y.min()) / (y.max() - y.min() + 1e-10))

        # Plot each line
        x_positions = np.arange(len(param_names))
        for i in range(len(X_norm)):
            ax.plot(x_positions, X_norm[i], c=colors[i], alpha=0.5, linewidth=1)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_ylabel("Normalized Value")
        ax.set_title("Parallel Coordinates")

        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=y.min(), vmax=y.max()),
        )
        plt.colorbar(sm, ax=ax, label="Score")

        plt.tight_layout()
        return ax
