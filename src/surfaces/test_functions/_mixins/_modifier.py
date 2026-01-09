# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Modifier management mixin for test functions."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from surfaces.modifiers import BaseModifier


class ModifierMixin:
    """Mixin providing modifier management for test functions.

    Modifiers transform function values during evaluation. Examples include
    noise (GaussianNoise, UniformNoise) and delays (DelayModifier).

    Attributes
    ----------
    _modifiers : list of BaseModifier
        List of modifiers to apply to function evaluations.

    Notes
    -----
    This mixin expects the following attributes/methods to be present:
    - self.pure_objective_function: callable
    - self.objective: str ("minimize" or "maximize")
    - self._normalize_input(params, **kwargs): method
    """

    _modifiers: List[BaseModifier]

    def _init_modifiers(self, modifiers: Optional[List[BaseModifier]] = None) -> None:
        """Initialize modifier list.

        Parameters
        ----------
        modifiers : list of BaseModifier, optional
            List of modifiers to apply to function evaluations.
        """
        self._modifiers = modifiers if modifiers is not None else []

    def true_value(
        self,
        params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None,
        **kwargs,
    ) -> float:
        """Evaluate the function without modifiers.

        Returns the true (deterministic) function value, bypassing any
        configured modifiers. Useful for analysis and comparison.

        This method does not update search_data, n_evaluations, or callbacks.
        It also ignores memory caching.

        Parameters
        ----------
        params : dict, array, list, or tuple
            Parameter values to evaluate.
        **kwargs : dict
            Parameters as keyword arguments.

        Returns
        -------
        float
            The true function value without modifiers.

        Examples
        --------
        >>> from surfaces.modifiers import GaussianNoise
        >>> func = SphereFunction(
        ...     n_dim=2,
        ...     modifiers=[GaussianNoise(sigma=0.1, seed=42)]
        ... )
        >>> modified = func([1.0, 2.0])
        >>> true = func.true_value([1.0, 2.0])
        >>> print(f"Difference: {modified - true:.4f}")
        """
        params = self._normalize_input(params, **kwargs)
        raw_value = self.pure_objective_function(params)
        if self.objective == "maximize":
            return -raw_value
        return raw_value

    def reset_modifiers(self) -> None:
        """Reset all modifiers' internal state.

        Resets evaluation counters, random states, and any other
        stateful components in the modifiers list.
        """
        for modifier in self._modifiers:
            modifier.reset()

    @property
    def modifiers(self) -> List[BaseModifier]:
        """The list of modifiers for this function."""
        return self._modifiers
