# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for multi-objective test functions.

This intermediate base extends :class:`BaseTestFunction` with multi-objective
specifics: vector-valued returns, Pareto front/set methods, and modifier
handling that preserves the objective vector.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from ..._base_test_function import BaseTestFunction


class BaseMultiObjectiveTestFunction(BaseTestFunction):
    """Base class for multi-objective test functions.

    Multi-objective functions return a vector of objective values instead of
    a scalar. The goal is typically to find the Pareto front -- the set of
    solutions where no objective can be improved without worsening another.

    Parameters
    ----------
    n_dim : int, default=2
        Number of input dimensions.
    objective : str or list of str, default="minimize"
        Optimization direction(s). A single string is broadcast to all
        objectives. A list must have length ``n_objectives``.
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.
        Note: Value-modifying modifiers (like noise) are not supported for
        multi-objective functions. Only side-effect modifiers (like delay)
        work correctly.
    memory : bool, default=False
        If True, caches evaluated positions.
    collect_data : bool, default=True
        If True, collects evaluation data.
    callbacks : callable or list of callables, optional
        Function(s) called after each evaluation.
    catch_errors : dict, optional
        Dictionary mapping exception types to return values.

    Attributes
    ----------
    n_objectives : int
        Number of objectives (set by subclass, default 2).
    n_dim : int
        Number of input dimensions.
    """

    n_objectives: int = 2
    default_size: int = 1000

    _spec: Dict[str, Any] = {
        "n_objectives": 2,
        "continuous": True,
        "differentiable": True,
        "convex": False,
        "scalable": True,
        "default_bounds": (0.0, 1.0),
    }

    def __init__(
        self,
        n_dim: int = 2,
        objective: Union[str, List[str]] = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks=None,
        catch_errors=None,
    ):
        self.n_dim = n_dim

        # Normalize objective to a list of per-objective directions
        if isinstance(objective, str):
            if objective not in ("minimize", "maximize"):
                raise ValueError(f"objective must be 'minimize' or 'maximize', got '{objective}'")
            objective = [objective] * self.n_objectives
        else:
            if len(objective) != self.n_objectives:
                raise ValueError(
                    f"objective list must have length {self.n_objectives}, " f"got {len(objective)}"
                )
            for obj in objective:
                if obj not in ("minimize", "maximize"):
                    raise ValueError(
                        f"each objective must be 'minimize' or 'maximize', got '{obj}'"
                    )

        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

    # -----------------------------------------------------------------
    # Direction: per-element negation for maximize objectives
    # -----------------------------------------------------------------

    def _apply_direction(self, value):
        """Negate objective components where direction is 'maximize'.

        Handles both single evaluations (1D) and batch evaluations (2D).
        """
        mask = np.array([obj == "maximize" for obj in self.objective])
        if mask.any():
            value = np.array(value, dtype=float)
            if value.ndim == 2:
                value[:, mask] = -value[:, mask]
            else:
                value[mask] = -value[mask]
        return value

    # -----------------------------------------------------------------
    # Search space
    # -----------------------------------------------------------------

    def _default_search_space(self) -> Dict[str, Any]:
        """Build search space from default_bounds and n_dim."""
        min_val, max_val = self.spec.default_bounds
        return self._create_search_space(min=min_val, max=max_val, size=self.default_size)

    def _create_search_space(
        self,
        min: float = 0.0,
        max: float = 1.0,
        size: int = 1000,
    ) -> Dict[str, Any]:
        """Create search space for the function."""
        search_space = {}
        dim_size = int(size ** (1 / self.n_dim))

        for dim in range(self.n_dim):
            dim_str = f"x{dim}"
            step_size = (max - min) / dim_size
            values = np.arange(min, max, step_size)
            search_space[dim_str] = values

        return search_space

    # -----------------------------------------------------------------
    # Modifier handling: side-effects only, preserve objective vector
    # -----------------------------------------------------------------

    def _apply_modifiers(self, raw_value, params):
        """Apply modifiers for side-effects only.

        The first objective value is passed through the modifier pipeline
        (triggering side-effects like delays), but the original objective
        vector is returned unchanged.
        """
        context = {}
        value = raw_value[0]
        for modifier in self._modifiers:
            value = modifier.apply(value, params, context)
        return raw_value

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dict to numpy array."""
        return np.array([params[f"x{i}"] for i in range(self.n_dim)])

    # -----------------------------------------------------------------
    # Pareto front / set (template method pattern)
    # -----------------------------------------------------------------

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        """Generate points on the theoretical Pareto front.

        Parameters
        ----------
        n_points : int, default=100
            Number of points to generate on the Pareto front.

        Returns
        -------
        np.ndarray
            Array of shape (n_points, n_objectives) containing points
            on the Pareto front.
        """
        return self._pareto_front(n_points)

    def _pareto_front(self, n_points: int) -> np.ndarray:
        """Compute the Pareto front by evaluating the Pareto set.

        Default implementation evaluates ``_pareto_set`` points through
        ``_objective``.  Subclasses may override for an analytical shortcut.
        """
        x_samples = self._pareto_set(n_points)
        front = np.zeros((n_points, self.n_objectives))
        for i, x in enumerate(x_samples):
            params = {f"x{j}": x[j] for j in range(self.n_dim)}
            front[i] = self._objective(params)
        return front

    def pareto_set(self, n_points: int = 100) -> np.ndarray:
        """Generate points in the Pareto set (decision space).

        Parameters
        ----------
        n_points : int, default=100
            Number of points to generate in the Pareto set.

        Returns
        -------
        np.ndarray
            Array of shape (n_points, n_dim) containing points
            in the Pareto set.
        """
        return self._pareto_set(n_points)

    def _pareto_set(self, n_points: int) -> np.ndarray:
        """Compute the Pareto set. Override in subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _pareto_set(self, n_points)"
        )


# Backward compatibility alias
MultiObjectiveFunction = BaseMultiObjectiveTestFunction
