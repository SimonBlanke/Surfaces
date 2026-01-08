# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for CEC 2006 constrained benchmark functions."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ....algebraic._base_algebraic_function import AlgebraicFunction


class CEC2006Function(AlgebraicFunction):
    """Base class for CEC 2006 constrained benchmark functions.

    CEC 2006 is a benchmark suite for constrained real-parameter optimization.
    Unlike CEC 2013/2014/2017, these functions:
    - Have fixed dimensions (not scalable)
    - Include inequality and equality constraints
    - Use the original function forms (no shift/rotation)
    - Have problem-specific bounds

    Constraints are handled using the exterior penalty method by default,
    converting constrained problems into unconstrained ones.

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    penalty_coefficient : float, default=1e6
        Coefficient for constraint violation penalties.
    equality_tolerance : float, default=1e-4
        Tolerance for equality constraints. Equality h(x)=0 is converted
        to |h(x)| - tolerance <= 0.

    Attributes
    ----------
    func_id : int
        Function ID (1-24 for G01-G24).
    n_dim : int
        Number of dimensions (fixed per function).
    n_linear_ineq : int
        Number of linear inequality constraints.
    n_nonlinear_eq : int
        Number of nonlinear equality constraints.
    n_nonlinear_ineq : int
        Number of nonlinear inequality constraints.
    f_global : float
        Best known objective value.
    x_global : np.ndarray
        Best known solution.

    References
    ----------
    Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M.,
    Suganthan, P. N., Coello, C. A. C., & Deb, K. (2006).
    Problem definitions and evaluation criteria for the CEC 2006
    special session on constrained real-parameter optimization.
    Technical Report, Nanyang Technological University, Singapore.
    """

    _spec = {
        "func_id": None,
        "default_bounds": None,  # Variable-specific bounds
        "continuous": True,
        "differentiable": True,
        "scalable": False,  # Fixed dimensions
        "constrained": True,
    }

    # Subclasses define these
    _n_dim: int = 0
    _n_linear_ineq: int = 0
    _n_nonlinear_eq: int = 0
    _n_nonlinear_ineq: int = 0
    _f_global: float = 0.0
    _x_global: Optional[np.ndarray] = None
    _variable_bounds: List[Tuple[float, float]] = []

    def __init__(
        self,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        penalty_coefficient: float = 1e6,
        equality_tolerance: float = 1e-4,
    ) -> None:
        self.penalty_coefficient = penalty_coefficient
        self.equality_tolerance = equality_tolerance
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

    @property
    def n_dim(self) -> int:
        """Number of dimensions (fixed for this function)."""
        return self._n_dim

    @property
    def func_id(self) -> int:
        """Function ID within the CEC 2006 suite."""
        return self._spec.get("func_id", 0)

    @property
    def n_linear_ineq(self) -> int:
        """Number of linear inequality constraints."""
        return self._n_linear_ineq

    @property
    def n_nonlinear_eq(self) -> int:
        """Number of nonlinear equality constraints."""
        return self._n_nonlinear_eq

    @property
    def n_nonlinear_ineq(self) -> int:
        """Number of nonlinear inequality constraints."""
        return self._n_nonlinear_ineq

    @property
    def n_constraints(self) -> int:
        """Total number of constraints."""
        return self._n_linear_ineq + self._n_nonlinear_eq + self._n_nonlinear_ineq

    @property
    def f_global(self) -> float:
        """Best known objective value."""
        return self._f_global

    @property
    def x_global(self) -> Optional[np.ndarray]:
        """Best known solution."""
        return self._x_global

    @property
    def variable_bounds(self) -> List[Tuple[float, float]]:
        """Bounds for each variable as (min, max) pairs."""
        return self._variable_bounds

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space based on variable bounds."""
        search_space_ = {}
        total_size = 10000
        dim_size = max(2, int(total_size ** (1 / self.n_dim)))

        for i, (lb, ub) in enumerate(self.variable_bounds):
            step_size = (ub - lb) / dim_size
            values = np.arange(lb, ub + step_size / 2, step_size)
            search_space_[f"x{i}"] = values

        return search_space_

    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dict to numpy array."""
        return np.array([params[f"x{i}"] for i in range(self.n_dim)])

    # =========================================================================
    # Constraint methods (to be overridden by subclasses)
    # =========================================================================

    def inequality_constraints(self, x: np.ndarray) -> List[float]:
        """Evaluate inequality constraints g_i(x) <= 0.

        Parameters
        ----------
        x : np.ndarray
            Decision variable vector.

        Returns
        -------
        list of float
            Constraint values. Feasible if all <= 0.
        """
        return []

    def equality_constraints(self, x: np.ndarray) -> List[float]:
        """Evaluate equality constraints h_i(x) = 0.

        Parameters
        ----------
        x : np.ndarray
            Decision variable vector.

        Returns
        -------
        list of float
            Constraint values. Feasible if all == 0 (within tolerance).
        """
        return []

    def constraint_violations(self, x: np.ndarray) -> Tuple[List[float], List[float]]:
        """Calculate constraint violations.

        Parameters
        ----------
        x : np.ndarray
            Decision variable vector.

        Returns
        -------
        tuple of (list of float, list of float)
            (inequality_violations, equality_violations).
            Zero means constraint is satisfied.
        """
        ineq = [max(0, g) for g in self.inequality_constraints(x)]
        # Equality: |h(x)| - tolerance, clipped to >= 0
        eq = [max(0, abs(h) - self.equality_tolerance) for h in self.equality_constraints(x)]
        return ineq, eq

    def is_feasible(self, x: np.ndarray) -> bool:
        """Check if a solution satisfies all constraints.

        Parameters
        ----------
        x : np.ndarray
            Decision variable vector.

        Returns
        -------
        bool
            True if all constraints are satisfied.
        """
        ineq, eq = self.constraint_violations(x)
        return all(v == 0 for v in ineq + eq)

    def penalty(self, x: np.ndarray) -> float:
        """Calculate total penalty for constraint violations.

        Uses exterior penalty method with squared violations.

        Parameters
        ----------
        x : np.ndarray
            Decision variable vector.

        Returns
        -------
        float
            Penalty value.
        """
        ineq, eq = self.constraint_violations(x)
        total = sum(v**2 for v in ineq) + sum(v**2 for v in eq)
        return self.penalty_coefficient * total

    def raw_objective(self, x: np.ndarray) -> float:
        """Evaluate the raw objective function without penalties.

        Override in subclasses.

        Parameters
        ----------
        x : np.ndarray
            Decision variable vector.

        Returns
        -------
        float
            Raw objective value.
        """
        raise NotImplementedError("Subclasses must implement raw_objective")

    def _create_objective_function(self) -> None:
        """Create objective function with penalty for constraint violations."""

        def penalized_objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self.raw_objective(x) + self.penalty(x)

        self.pure_objective_function = penalized_objective

    # =========================================================================
    # Batch evaluation methods
    # =========================================================================

    def _batch_raw_objective(self, X: ArrayLike) -> ArrayLike:
        """Compute raw objective for batch of points.

        Override in subclasses for vectorized implementation.

        Parameters
        ----------
        X : ArrayLike
            Input batch of shape (n_points, n_dim).

        Returns
        -------
        ArrayLike
            Raw objective values of shape (n_points,).
        """
        raise NotImplementedError("Subclasses must implement _batch_raw_objective")

    def _batch_inequality_constraints(self, X: ArrayLike) -> ArrayLike:
        """Compute inequality constraints for batch of points.

        Override in subclasses for vectorized implementation.

        Parameters
        ----------
        X : ArrayLike
            Input batch of shape (n_points, n_dim).

        Returns
        -------
        ArrayLike
            Constraint values of shape (n_points, n_ineq_constraints).
        """
        xp = get_array_namespace(X)
        n_ineq = self._n_linear_ineq + self._n_nonlinear_ineq
        return xp.zeros((X.shape[0], n_ineq))

    def _batch_equality_constraints(self, X: ArrayLike) -> ArrayLike:
        """Compute equality constraints for batch of points.

        Override in subclasses for vectorized implementation.

        Parameters
        ----------
        X : ArrayLike
            Input batch of shape (n_points, n_dim).

        Returns
        -------
        ArrayLike
            Constraint values of shape (n_points, n_eq_constraints).
        """
        xp = get_array_namespace(X)
        return xp.zeros((X.shape[0], self._n_nonlinear_eq))

    def _batch_penalty(self, X: ArrayLike) -> ArrayLike:
        """Compute penalty for batch of points.

        Parameters
        ----------
        X : ArrayLike
            Input batch of shape (n_points, n_dim).

        Returns
        -------
        ArrayLike
            Penalty values of shape (n_points,).
        """
        xp = get_array_namespace(X)

        # Inequality violations: max(0, g)^2
        G = self._batch_inequality_constraints(X)
        ineq_violations = xp.sum(xp.maximum(G, 0.0) ** 2, axis=1)

        # Equality violations: max(0, |h| - tol)^2
        H = self._batch_equality_constraints(X)
        eq_violations = xp.sum(xp.maximum(xp.abs(H) - self.equality_tolerance, 0.0) ** 2, axis=1)

        return self.penalty_coefficient * (ineq_violations + eq_violations)

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Compute penalized objective for batch of points.

        Parameters
        ----------
        X : ArrayLike
            Input batch of shape (n_points, n_dim).

        Returns
        -------
        ArrayLike
            Penalized objective values of shape (n_points,).
        """
        return self._batch_raw_objective(X) + self._batch_penalty(X)
