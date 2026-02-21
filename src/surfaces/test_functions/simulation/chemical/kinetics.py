# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Chemical kinetics test functions based on ODE systems."""

from typing import Any, Dict

import numpy as np

from ..dynamics._base_ode import ODESimulationFunction


class ConsecutiveReactionFunction(ODESimulationFunction):
    """Consecutive chemical reaction optimization problem.

    Models the reaction scheme A -> B -> C with first-order kinetics:

        dA/dt = -k1 * A
        dB/dt = k1 * A - k2 * B
        dC/dt = k2 * B

    The optimization objective is to find rate constants k1 and k2 that
    maximize the yield of intermediate product B at a target time.

    This is a classic problem in chemical engineering: B is often the
    desired product, but it gets consumed to form C if reaction continues.

    Parameters
    ----------
    A0 : float, default=1.0
        Initial concentration of A.
    B0 : float, default=0.0
        Initial concentration of B.
    C0 : float, default=0.0
        Initial concentration of C.
    t_span : tuple, default=(0, 10)
        Time interval for simulation.
    target_time : float, optional
        Time at which to evaluate B concentration. If None, uses t_span[1].
    objective_type : str, default="max_B"
        Type of objective:
        - "max_B": Maximize B concentration at target_time
        - "max_B_integral": Maximize time-integrated B (total B produced)
        - "selectivity": Maximize B/(B+C) ratio at target_time
    **kwargs
        Additional arguments passed to ODESimulationFunction.

    Search Space
    ------------
    k1 : float in [0.01, 5.0]
        Rate constant for A -> B reaction.
    k2 : float in [0.01, 5.0]
        Rate constant for B -> C reaction.

    Notes
    -----
    Analytical optimum for max_B at time t:
        k1_opt / k2_opt = exp((k1_opt - k2_opt) * t)

    For max_B objective, the landscape has a clear global optimum but
    with a curved ridge structure.

    Examples
    --------
    >>> func = ConsecutiveReactionFunction(target_time=2.0)
    >>> result = func({"k1": 1.0, "k2": 0.5})

    References
    ----------
    .. [1] Levenspiel, O. (1999). Chemical Reaction Engineering, 3rd ed.
    """

    _spec = {
        "simulation_based": True,
        "expensive": False,
        "continuous": True,
        "ode_based": True,
        "n_dim": 2,
        "unimodal": True,
        "domain": "chemistry",
    }

    def __init__(
        self,
        A0: float = 1.0,
        B0: float = 0.0,
        C0: float = 0.0,
        t_span: tuple = (0, 10),
        target_time: float = None,
        objective_type: str = "max_B",
        **kwargs,
    ) -> None:
        self.A0 = A0
        self.B0 = B0
        self.C0 = C0
        self.target_time = target_time if target_time is not None else t_span[1]
        self.objective_type = objective_type
        t_eval = np.linspace(t_span[0], t_span[1], 500)
        super().__init__(t_span=t_span, t_eval=t_eval, **kwargs)

    @property
    def search_space(self) -> Dict[str, np.ndarray]:
        """2D search space for rate constants."""
        return {
            "k1": np.linspace(0.01, 5.0, 100),
            "k2": np.linspace(0.01, 5.0, 100),
        }

    def _get_initial_conditions(self) -> np.ndarray:
        """Return initial concentrations [A, B, C]."""
        return np.array([self.A0, self.B0, self.C0])

    def _ode_system(self, t: float, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Consecutive reaction ODE system.

        dA/dt = -k1 * A
        dB/dt = k1 * A - k2 * B
        dC/dt = k2 * B
        """
        A, B, C = y
        k1 = params["k1"]
        k2 = params["k2"]

        dA_dt = -k1 * A
        dB_dt = k1 * A - k2 * B
        dC_dt = k2 * B

        return np.array([dA_dt, dB_dt, dC_dt])

    def _compute_objective(self, t: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        """Compute objective from concentration trajectories."""
        _A = y[0, :]  # noqa: F841 - extracted for completeness
        B = y[1, :]
        C = y[2, :]

        # Check for numerical issues
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return float("inf")

        if self.objective_type == "max_B":
            # Find B concentration at target_time (interpolate)
            B_at_target = np.interp(self.target_time, t, B)
            # Return negative for minimization (we want to maximize B)
            return -B_at_target

        elif self.objective_type == "max_B_integral":
            # Maximize time-integrated B concentration
            B_integral = np.trapz(B, t)
            return -B_integral

        elif self.objective_type == "selectivity":
            # Maximize B/(B+C) selectivity at target_time
            B_at_target = np.interp(self.target_time, t, B)
            C_at_target = np.interp(self.target_time, t, C)

            if B_at_target + C_at_target < 1e-10:
                return float("inf")

            selectivity = B_at_target / (B_at_target + C_at_target)
            return -selectivity  # Negative for minimization

        else:
            raise ValueError(f"Unknown objective_type: {self.objective_type}")
