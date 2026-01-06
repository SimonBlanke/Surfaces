# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Population dynamics test functions based on ODE systems."""

from typing import Any, Dict

import numpy as np

from ._base_ode import ODESimulationFunction


class LotkaVolterraFunction(ODESimulationFunction):
    """Lotka-Volterra predator-prey population dynamics.

    The Lotka-Volterra equations model the dynamics of predator-prey interactions:

        dx/dt = alpha * x - beta * x * y    (prey growth - predation)
        dy/dt = delta * x * y - gamma * y   (predator growth - death)

    where x is prey population and y is predator population.

    The optimization objective is to find parameters that minimize population
    oscillation variance, leading to stable coexistence.

    Parameters
    ----------
    x0 : float, default=10.0
        Initial prey population.
    y0 : float, default=5.0
        Initial predator population.
    t_span : tuple, default=(0, 50)
        Time interval for simulation.
    objective_type : str, default="variance"
        Type of objective:
        - "variance": Minimize total population variance (find stability)
        - "prey_survival": Maximize minimum prey population
        - "balance": Minimize difference between final predator/prey ratio and 1
    **kwargs
        Additional arguments passed to ODESimulationFunction.

    Search Space
    ------------
    alpha : float in [0.1, 2.0]
        Prey birth rate.
    beta : float in [0.01, 0.5]
        Predation rate (prey death per predator encounter).
    gamma : float in [0.1, 2.0]
        Predator death rate.
    delta : float in [0.01, 0.5]
        Predator reproduction rate per prey consumed.

    Global Optimum
    --------------
    The global optimum depends on objective_type. For "variance", the optimum
    occurs when parameters satisfy the equilibrium condition:
        x* = gamma/delta, y* = alpha/beta

    Examples
    --------
    >>> func = LotkaVolterraFunction()
    >>> # Parameters near equilibrium should give low variance
    >>> result = func({"alpha": 0.5, "beta": 0.1, "gamma": 0.5, "delta": 0.1})

    References
    ----------
    .. [1] Lotka, A.J. (1925). Elements of Physical Biology.
    .. [2] Volterra, V. (1926). Variazioni e fluttuazioni del numero d'individui
           in specie animali conviventi.
    """

    _spec = {
        "simulation_based": True,
        "expensive": False,
        "continuous": True,
        "ode_based": True,
        "n_dim": 4,
        "multimodal": True,
        "domain": "ecology",
    }

    def __init__(
        self,
        x0: float = 10.0,
        y0: float = 5.0,
        t_span: tuple = (0, 50),
        objective_type: str = "variance",
        **kwargs,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.objective_type = objective_type
        # Use many evaluation points for accurate variance calculation
        t_eval = np.linspace(t_span[0], t_span[1], 500)
        super().__init__(t_span=t_span, t_eval=t_eval, **kwargs)

    @property
    def search_space(self) -> Dict[str, np.ndarray]:
        """4D search space for Lotka-Volterra parameters."""
        return {
            "alpha": np.linspace(0.1, 2.0, 100),  # Prey birth rate
            "beta": np.linspace(0.01, 0.5, 100),  # Predation rate
            "gamma": np.linspace(0.1, 2.0, 100),  # Predator death rate
            "delta": np.linspace(0.01, 0.5, 100),  # Predator reproduction
        }

    def _get_initial_conditions(self) -> np.ndarray:
        """Return initial populations [prey, predator]."""
        return np.array([self.x0, self.y0])

    def _ode_system(self, t: float, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Lotka-Volterra ODE system.

        dx/dt = alpha*x - beta*x*y
        dy/dt = delta*x*y - gamma*y
        """
        x, y_pop = y  # prey, predator
        alpha = params["alpha"]
        beta = params["beta"]
        gamma = params["gamma"]
        delta = params["delta"]

        dx_dt = alpha * x - beta * x * y_pop
        dy_dt = delta * x * y_pop - gamma * y_pop

        return np.array([dx_dt, dy_dt])

    def _compute_objective(self, t: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        """Compute objective from population trajectories."""
        prey = y[0, :]
        predator = y[1, :]

        # Check for extinction or explosion
        if np.any(prey <= 0) or np.any(predator <= 0):
            return float("inf")
        if np.any(prey > 1e6) or np.any(predator > 1e6):
            return float("inf")

        if self.objective_type == "variance":
            # Minimize total population variance (stability objective)
            # Use coefficient of variation to normalize
            prey_cv = np.std(prey) / np.mean(prey)
            predator_cv = np.std(predator) / np.mean(predator)
            return prey_cv + predator_cv

        elif self.objective_type == "prey_survival":
            # Maximize minimum prey population (negative for minimization)
            return -np.min(prey)

        elif self.objective_type == "balance":
            # Minimize deviation from balanced populations
            ratio = predator / prey
            return np.std(ratio) + abs(np.mean(ratio) - 1.0)

        else:
            raise ValueError(f"Unknown objective_type: {self.objective_type}")
