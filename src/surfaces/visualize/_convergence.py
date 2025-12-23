# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Convergence plot for optimization history."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ..test_functions._base_test_function import BaseTestFunction

from ._errors import MissingDataError
from ._utils import check_plotly, go


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
    check_plotly()

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
