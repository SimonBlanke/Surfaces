# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for engineering design optimization test functions."""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .._base_test_function import BaseTestFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class EngineeringFunction(BaseTestFunction):
    """Base class for real-world engineering design optimization problems.

    Engineering functions represent practical design optimization problems
    from domains like structural mechanics, manufacturing, and mechanical
    engineering. Unlike purely mathematical test functions, these problems
    have physical meaning and constraints derived from engineering principles.

    Most engineering problems are inherently constrained. This base class
    provides infrastructure for handling constraints via penalty methods,
    converting constrained problems into unconstrained ones suitable for
    general-purpose optimizers.

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.
    penalty_coefficient : float, default=1e6
        Coefficient for constraint violation penalties. Higher values
        enforce constraints more strictly but may create steep gradients.

    Attributes
    ----------
    n_dim : int
        Number of design variables.
    variable_names : list of str
        Names of design variables (e.g., ["thickness", "radius"]).
    variable_bounds : list of tuple
        Bounds for each variable as (min, max) pairs.

    Notes
    -----
    Constraint handling uses the exterior penalty method:

    .. math::

        F(x) = f(x) + r \\sum_{i} \\max(0, g_i(x))^2

    where f(x) is the objective, g_i(x) are inequality constraints
    (g_i(x) <= 0 is feasible), and r is the penalty coefficient.
    """

    _spec = {
        "default_bounds": None,  # Engineering functions have variable-specific bounds
        "continuous": True,
        "differentiable": True,
        "constrained": True,
    }

    default_size: int = 10000

    # Subclasses should define these
    variable_names: List[str] = []
    variable_bounds: List[Tuple[float, float]] = []

    def __init__(
        self,
        objective: str = "minimize",
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        noise: Optional["BaseNoise"] = None,
        penalty_coefficient: float = 1e6,
    ) -> None:
        self.penalty_coefficient = penalty_coefficient
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)

    @property
    def n_dim(self) -> int:
        """Number of design variables."""
        return len(self.variable_names)

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space based on variable bounds."""
        search_space_ = {}
        total_size = self.default_size
        dim_size = int(total_size ** (1 / self.n_dim))

        for i, (name, (lb, ub)) in enumerate(zip(self.variable_names, self.variable_bounds)):
            step_size = (ub - lb) / dim_size
            values = np.arange(lb, ub, step_size)
            search_space_[name] = values

        return search_space_

    def _get_values(self, params: Dict[str, Any]) -> np.ndarray:
        """Extract variable values from params dict in order."""
        return np.array([params[name] for name in self.variable_names])

    def constraints(self, params: Dict[str, Any]) -> List[float]:
        """Evaluate constraint functions.

        Override in subclasses to define problem-specific constraints.
        Each constraint g_i should be formulated such that g_i <= 0 is feasible.

        Parameters
        ----------
        params : dict
            Design variable values.

        Returns
        -------
        list of float
            Constraint function values. Negative or zero means feasible.
        """
        return []

    def constraint_violations(self, params: Dict[str, Any]) -> List[float]:
        """Calculate constraint violations (positive values only).

        Parameters
        ----------
        params : dict
            Design variable values.

        Returns
        -------
        list of float
            Violation amounts. Zero means constraint is satisfied.
        """
        return [max(0, g) for g in self.constraints(params)]

    def is_feasible(self, params: Dict[str, Any]) -> bool:
        """Check if a solution satisfies all constraints.

        Parameters
        ----------
        params : dict
            Design variable values.

        Returns
        -------
        bool
            True if all constraints are satisfied.
        """
        return all(g <= 0 for g in self.constraints(params))

    def penalty(self, params: Dict[str, Any]) -> float:
        """Calculate total penalty for constraint violations.

        Parameters
        ----------
        params : dict
            Design variable values.

        Returns
        -------
        float
            Penalty value (sum of squared violations times coefficient).
        """
        violations = self.constraint_violations(params)
        return self.penalty_coefficient * sum(v**2 for v in violations)

    def raw_objective(self, params: Dict[str, Any]) -> float:
        """Evaluate the raw objective function without penalties.

        Override in subclasses to define the engineering objective.

        Parameters
        ----------
        params : dict
            Design variable values.

        Returns
        -------
        float
            Raw objective function value.
        """
        raise NotImplementedError("Subclasses must implement raw_objective")

    def _create_objective_function(self) -> None:
        """Create objective function with penalty for constraint violations."""

        def penalized_objective(params: Dict[str, Any]) -> float:
            return self.raw_objective(params) + self.penalty(params)

        self.pure_objective_function = penalized_objective
