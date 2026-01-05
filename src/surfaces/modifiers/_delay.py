# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Delay modifier for simulating expensive function evaluations."""

import time
from typing import Any, Dict

from ._base_modifier import BaseModifier


class DelayModifier(BaseModifier):
    """Adds artificial delay to function evaluations.

    Useful for simulating expensive function evaluations or testing
    optimizer behavior under time constraints.

    Parameters
    ----------
    delay : float
        Delay in seconds to add to each evaluation.

    Examples
    --------
    >>> from surfaces.modifiers import DelayModifier
    >>> from surfaces import SphereFunction
    >>> func = SphereFunction(
    ...     n_dim=2,
    ...     modifiers=[DelayModifier(delay=0.1)]
    ... )
    >>> # Each evaluation now takes at least 0.1 seconds
    """

    def __init__(self, delay: float):
        if delay < 0:
            raise ValueError(f"delay must be non-negative, got {delay}")
        self.delay = delay

    def apply(self, value: float, params: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Apply delay and return unchanged value.

        Parameters
        ----------
        value : float
            The function value (unchanged).
        params : dict
            The input parameters (unused).
        context : dict
            Context information (unused).

        Returns
        -------
        float
            The original value unchanged.
        """
        time.sleep(self.delay)
        return value

    def __repr__(self) -> str:
        """String representation.

        Returns
        -------
        str
            String representation.
        """
        return f"DelayModifier(delay={self.delay})"
