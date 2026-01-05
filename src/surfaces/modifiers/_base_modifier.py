# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for function modifiers."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModifier(ABC):
    """Base class for modifiers that can be applied to test functions.

    Modifiers transform function values or add effects like noise, delays,
    rotations, constraints, etc. They are applied in a pipeline according
    to their order in the modifiers list.

    Examples
    --------
    >>> from surfaces.modifiers import BaseModifier
    >>> class MyModifier(BaseModifier):
    ...     def apply(self, value, params, context):
    ...         return value * 2
    >>> modifier = MyModifier()
    >>> modifier.apply(5.0, {"x0": 1.0}, {"evaluation_count": 0})
    10.0
    """

    @abstractmethod
    def apply(self, value: float, params: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Apply the modifier to a function value.

        Parameters
        ----------
        value : float
            The function value to modify.
        params : dict
            The input parameters that produced this value.
        context : dict
            Additional context information, may contain:
            - evaluation_count: int - Number of evaluations performed
            - best_score: float - Best score found so far
            - search_data: list - History of evaluations

        Returns
        -------
        float
            The modified function value.
        """
        pass

    def reset(self) -> None:
        """Reset the modifier's internal state.

        Override this method if your modifier maintains state
        (e.g., evaluation counters, random number generators).
        """
        pass
