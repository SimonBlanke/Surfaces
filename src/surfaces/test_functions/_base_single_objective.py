# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for single-objective test functions.

This intermediate base adds the ``objective`` parameter (minimize/maximize)
and all scalar-specific logic (negation, best-score tracking, ``pure()``).
All single-objective function hierarchies (algebraic, ML, simulation,
engineering, custom) inherit from this class.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from surfaces._array_utils import ArrayLike
from surfaces.modifiers import BaseModifier

from ._base_test_function import BaseTestFunction


class BaseSingleObjectiveTestFunction(BaseTestFunction):
    """Base class for single-objective test functions.

    Extends :class:`BaseTestFunction` with:
    - ``objective`` parameter ("minimize" / "maximize")
    - Negation in ``_evaluate`` for maximize
    - Scalar best-score tracking
    - ``pure()`` evaluation (no modifiers)
    - Batch negation for maximize

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.
    memory : bool, default=False
        If True, caches evaluated positions to avoid redundant computations.
    collect_data : bool, default=True
        If True, collects evaluation data.
    callbacks : callable or list of callables, optional
        Function(s) called after each evaluation with the record dict.
    catch_errors : dict, optional
        Dictionary mapping exception types to return values.
    """

    def __init__(
        self,
        objective="minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
    ):
        if objective not in ("minimize", "maximize"):
            raise ValueError(f"objective must be 'minimize' or 'maximize', got '{objective}'")
        self.objective = objective
        super().__init__(modifiers, memory, collect_data, callbacks, catch_errors)

    # -----------------------------------------------------------------
    # Evaluation: add negation for maximize
    # -----------------------------------------------------------------

    def _evaluate(self, params: Dict[str, Any]) -> float:
        """Evaluate with modifiers and negate for maximize."""
        result = super()._evaluate(params)
        if self.objective == "maximize":
            return -result
        return result

    # -----------------------------------------------------------------
    # Best-score tracking (scalar comparison)
    # -----------------------------------------------------------------

    def _update_best(self, params: Dict[str, Any], score: float) -> None:
        """Update best score/params using scalar comparison."""
        is_better = (
            self._best_score is None
            or (self.objective == "minimize" and score < self._best_score)
            or (self.objective == "maximize" and score > self._best_score)
        )
        if is_better:
            self._best_score = score
            self._best_params = params.copy()

    # -----------------------------------------------------------------
    # Pure evaluation (no modifiers, no recording)
    # -----------------------------------------------------------------

    def pure(
        self,
        params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None,
        **kwargs,
    ) -> float:
        """Evaluate the function without modifiers.

        Returns the true (deterministic) function value, bypassing any
        configured modifiers. Does not update search_data, n_evaluations,
        or callbacks. Ignores memory caching.

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
        """
        params = self._normalize_input(params, **kwargs)
        raw_value = self._objective(params)
        if self.objective == "maximize":
            return -raw_value
        return raw_value

    # -----------------------------------------------------------------
    # Batch: add negation for maximize
    # -----------------------------------------------------------------

    def batch(self, X: ArrayLike) -> ArrayLike:
        """Evaluate multiple parameter sets with objective-direction handling.

        Parameters
        ----------
        X : ArrayLike
            2D array of shape (n_points, n_dim).

        Returns
        -------
        ArrayLike
            1D array of shape (n_points,) with evaluation results.
        """
        result = super().batch(X)
        if self.objective == "maximize":
            result = -result
        return result
