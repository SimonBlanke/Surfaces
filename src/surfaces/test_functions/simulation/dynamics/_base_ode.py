# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for ODE-based simulation test functions."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .._base_simulation import SimulationFunction


class ODESimulationFunction(SimulationFunction):
    """Base class for ODE-based optimization test functions.

    This class provides a common interface for optimization problems defined
    by systems of ordinary differential equations (ODEs). The objective is
    computed by integrating the ODE system and extracting a scalar metric.

    Subclasses should implement:
    - `_ode_system(t, y, params)`: The ODE right-hand side dy/dt = f(t, y, params)
    - `_compute_objective(t, y, params)`: Extract objective from solution
    - `_get_initial_conditions()`: Return y0 for the ODE system

    Parameters
    ----------
    t_span : tuple of float, default=(0, 10)
        Time interval for integration (t_start, t_end).
    t_eval : array-like, optional
        Times at which to store the solution. If None, solver chooses.
    method : str, default="RK45"
        Integration method: "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA".
    rtol : float, default=1e-6
        Relative tolerance for the solver.
    atol : float, default=1e-9
        Absolute tolerance for the solver.
    **kwargs
        Additional arguments passed to SimulationFunction.

    Attributes
    ----------
    t_span : tuple
        Integration time interval.
    n_equations : int
        Number of equations in the ODE system.
    """

    _spec = {
        "simulation_based": True,
        "expensive": False,  # ODE simulations are typically fast
        "continuous": True,
        "ode_based": True,
    }

    requires: List[str] = []  # scipy is always available

    @property
    def search_space(self) -> Dict[str, np.ndarray]:
        """Search space for ODE parameters (override in subclasses)."""
        raise NotImplementedError("Subclasses must implement search_space property")

    def __init__(
        self,
        t_span: Tuple[float, float] = (0, 10),
        t_eval: Optional[np.ndarray] = None,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **kwargs,
    ) -> None:
        self.t_span = t_span
        self.t_eval = t_eval
        self.method = method
        self.rtol = rtol
        self.atol = atol
        super().__init__(**kwargs)

    @abstractmethod
    def _ode_system(self, t: float, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Define the ODE system dy/dt = f(t, y, params).

        Parameters
        ----------
        t : float
            Current time.
        y : ndarray
            Current state vector.
        params : dict
            Parameter values for this evaluation.

        Returns
        -------
        ndarray
            Derivatives dy/dt.
        """
        pass

    @abstractmethod
    def _get_initial_conditions(self) -> np.ndarray:
        """Return initial conditions y0 for the ODE system.

        Returns
        -------
        ndarray
            Initial state vector.
        """
        pass

    @abstractmethod
    def _compute_objective(self, t: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        """Compute objective value from ODE solution.

        Parameters
        ----------
        t : ndarray
            Time points of the solution.
        y : ndarray
            Solution array of shape (n_equations, n_times).
        params : dict
            Parameter values used.

        Returns
        -------
        float
            Objective function value.
        """
        pass

    def _setup_simulation(self) -> None:
        """Initialize ODE simulation (nothing to setup for scipy)."""
        pass

    def _run_simulation(self, params: Dict[str, Any]) -> Any:
        """Integrate the ODE system."""
        y0 = self._get_initial_conditions()

        # Create wrapper that includes params
        def ode_wrapper(t, y):
            return self._ode_system(t, y, params)

        solution = solve_ivp(
            ode_wrapper,
            self.t_span,
            y0,
            method=self.method,
            t_eval=self.t_eval,
            rtol=self.rtol,
            atol=self.atol,
        )

        if not solution.success:
            # Return inf for failed integrations
            return None

        return solution

    def _extract_objective(self, result: Any) -> float:
        """Extract objective from ODE solution."""
        if result is None:
            return float("inf")

        # Get the parameters from the last call (stored during _run_simulation)
        # We need to pass params to _compute_objective, so we store them
        return self._compute_objective(result.t, result.y, self._current_params)

    def _create_objective_function(self) -> None:
        """Create objective function with parameter passing."""
        self._check_dependencies()
        self._setup_simulation()

        def simulation_objective(params: Dict[str, Any]) -> float:
            self._current_params = params  # Store for _extract_objective
            result = self._run_simulation(params)
            return self._extract_objective(result)

        self.pure_objective_function = simulation_objective
