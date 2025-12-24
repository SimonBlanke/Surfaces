# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Custom errors with helpful suggestions for visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..test_functions._base_test_function import BaseTestFunction


class VisualizationError(Exception):
    """Base exception for visualization errors."""

    pass


class PlotCompatibilityError(VisualizationError):
    """Raised when a plot type is incompatible with a function.

    Provides helpful suggestions for alternative plots.
    """

    def __init__(
        self,
        plot_name: str,
        reason: str,
        func: Optional["BaseTestFunction"] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.plot_name = plot_name
        self.reason = reason
        self.func = func
        self.suggestions = suggestions or []

        message = self._build_message()
        super().__init__(message)

    def _build_message(self) -> str:
        """Build a helpful error message with suggestions."""
        lines = [
            f"Cannot create '{self.plot_name}' plot: {self.reason}",
            "",
        ]

        if self.suggestions:
            lines.append("Try instead:")
            for suggestion in self.suggestions:
                lines.append(f"  - {suggestion}")
            lines.append("")

        lines.append("Use available_plots(func) to see all compatible plots.")

        return "\n".join(lines)


class MissingDataError(VisualizationError):
    """Raised when required data (e.g., optimization history) is missing."""

    def __init__(self, plot_name: str, data_type: str):
        self.plot_name = plot_name
        self.data_type = data_type

        message = self._build_message()
        super().__init__(message)

    def _build_message(self) -> str:
        lines = [
            f"Cannot create '{self.plot_name}' plot: requires {self.data_type}",
            "",
            "To collect optimization history, use TrackedFunction:",
            "",
            "    from surfaces.visualize import TrackedFunction",
            "    tracked = TrackedFunction(func)",
            "    # ... run optimization with tracked ...",
            f"    plot_{self.plot_name}(func, history=tracked.history)",
        ]
        return "\n".join(lines)


class MissingDependencyError(VisualizationError):
    """Raised when visualization dependencies are not installed."""

    def __init__(self, missing_packages: List[str]):
        self.missing_packages = missing_packages

        message = self._build_message()
        super().__init__(message)

    def _build_message(self) -> str:
        packages = ", ".join(self.missing_packages)
        return (
            f"Visualization requires additional packages: {packages}\n"
            f"Install with: pip install surfaces[viz]"
        )


# Suggestion mappings for common scenarios
DIMENSION_ALTERNATIVES = {
    "surface": {
        "1d": ["multi_slice"],
        "nd": ["multi_slice", "fitness_distribution"],
    },
    "contour": {
        "1d": ["multi_slice"],
        "nd": ["multi_slice", "fitness_distribution"],
    },
    "heatmap": {
        "1d": ["multi_slice"],
        "nd": ["multi_slice", "fitness_distribution"],
    },
}


def get_alternative_suggestions(plot_name: str, func: "BaseTestFunction") -> List[str]:
    """Get alternative plot suggestions based on function characteristics.

    Args:
        plot_name: The originally requested plot.
        func: The test function.

    Returns:
        List of suggestion strings.
    """
    from ._compatibility import _get_function_dimensions

    suggestions = []
    func_dims = _get_function_dimensions(func)

    if plot_name in DIMENSION_ALTERNATIVES:
        alternatives = DIMENSION_ALTERNATIVES[plot_name]

        if func_dims == 1 and "1d" in alternatives:
            for alt in alternatives["1d"]:
                suggestions.append(f"plot_{alt}(func) - works with 1D functions")
        elif func_dims > 2 and "nd" in alternatives:
            for alt in alternatives["nd"]:
                suggestions.append(f"plot_{alt}(func) - works with {func_dims}D functions")

            # Add slice suggestion for 2D visualization
            suggestions.append(f"plot_{plot_name}(func, dims=['x0', 'x1']) - fix other dimensions")

    if not suggestions:
        suggestions.append("auto_plot(func) - automatically select best visualization")

    return suggestions
