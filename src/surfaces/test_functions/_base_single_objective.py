# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for single-objective test functions.

This intermediate base adds the ``objective`` parameter (minimize/maximize)
and all scalar-specific logic (negation, best-score tracking).
All single-objective function hierarchies (algebraic, ML, simulation,
engineering, custom) inherit from this class.
"""

from typing import Any, Dict, List, Optional

from surfaces.modifiers import BaseModifier

from ._base_test_function import BaseTestFunction


class BaseSingleObjectiveTestFunction(BaseTestFunction):
    """Base class for single-objective test functions.

    Extends :class:`BaseTestFunction` with:
    - ``objective`` parameter ("minimize" / "maximize")
    - ``_apply_direction``: negate for maximize
    - Scalar best-score tracking

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
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

    # -----------------------------------------------------------------
    # Direction: negate for maximize
    # -----------------------------------------------------------------

    def _apply_direction(self, value):
        """Negate the value when objective is 'maximize'."""
        if self.objective == "maximize":
            return -value
        return value

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
