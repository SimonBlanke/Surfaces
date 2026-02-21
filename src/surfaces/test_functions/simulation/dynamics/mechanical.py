# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Mechanical systems test functions based on ODE systems."""

from typing import Any, Dict

import numpy as np

from ._base_ode import ODESimulationFunction


class DampedOscillatorFunction(ODESimulationFunction):
    """Damped harmonic oscillator optimization problem.

    The damped oscillator is described by:

        m * x'' + c * x' + k * x = 0

    Converted to first-order system:
        dx/dt = v
        dv/dt = -(c/m) * v - (k/m) * x

    The optimization objective is to find damping (c) and stiffness (k)
    parameters that achieve desired dynamic behavior.

    Parameters
    ----------
    mass : float, default=1.0
        Mass of the oscillator.
    x0 : float, default=1.0
        Initial displacement.
    v0 : float, default=0.0
        Initial velocity.
    t_span : tuple, default=(0, 10)
        Time interval for simulation.
    target_settling_percent : float, default=0.02
        Settling threshold (fraction of initial displacement).
    objective_type : str, default="settling_time"
        Type of objective:
        - "settling_time": Minimize time to reach settling threshold
        - "critical_damping": Find critically damped parameters
        - "overshoot": Minimize maximum overshoot
    **kwargs
        Additional arguments passed to ODESimulationFunction.

    Search Space
    ------------
    damping : float in [0.1, 10.0]
        Damping coefficient c.
    stiffness : float in [0.1, 20.0]
        Spring stiffness k.

    Notes
    -----
    Critical damping occurs when c = 2 * sqrt(k * m).

    For default mass=1.0:
    - Underdamped: c < 2*sqrt(k)
    - Critically damped: c = 2*sqrt(k)
    - Overdamped: c > 2*sqrt(k)

    Examples
    --------
    >>> func = DampedOscillatorFunction(objective_type="critical_damping")
    >>> # For k=4, critical damping is c=4
    >>> result = func({"damping": 4.0, "stiffness": 4.0})

    References
    ----------
    .. [1] Thomson, W.T. (1993). Theory of Vibration with Applications.
    """

    _spec = {
        "simulation_based": True,
        "expensive": False,
        "continuous": True,
        "ode_based": True,
        "n_dim": 2,
        "unimodal": True,  # For critical_damping objective
        "domain": "mechanics",
    }

    def __init__(
        self,
        mass: float = 1.0,
        x0: float = 1.0,
        v0: float = 0.0,
        t_span: tuple = (0, 10),
        target_settling_percent: float = 0.02,
        objective_type: str = "settling_time",
        **kwargs,
    ) -> None:
        self.mass = mass
        self.x0 = x0
        self.v0 = v0
        self.target_settling_percent = target_settling_percent
        self.objective_type = objective_type
        # Dense time points for accurate settling time detection
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        super().__init__(t_span=t_span, t_eval=t_eval, **kwargs)

    @property
    def search_space(self) -> Dict[str, np.ndarray]:
        """2D search space for oscillator parameters."""
        return {
            "damping": np.linspace(0.1, 10.0, 100),
            "stiffness": np.linspace(0.1, 20.0, 100),
        }

    def _get_initial_conditions(self) -> np.ndarray:
        """Return initial conditions [position, velocity]."""
        return np.array([self.x0, self.v0])

    def _ode_system(self, t: float, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Damped oscillator ODE system.

        dx/dt = v
        dv/dt = -(c/m)*v - (k/m)*x
        """
        x, v = y
        c = params["damping"]
        k = params["stiffness"]
        m = self.mass

        dx_dt = v
        dv_dt = -(c / m) * v - (k / m) * x

        return np.array([dx_dt, dv_dt])

    def _compute_objective(self, t: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        """Compute objective from oscillator trajectory."""
        x = y[0, :]  # Position
        c = params["damping"]
        k = params["stiffness"]

        if self.objective_type == "settling_time":
            # Find time when displacement stays below threshold
            threshold = self.target_settling_percent * abs(self.x0)
            settled_mask = np.abs(x) < threshold

            if not np.any(settled_mask):
                # Never settled within simulation time
                return self.t_span[1] * 2  # Penalty

            # Find first time it settles and stays settled
            for i in range(len(settled_mask)):
                if settled_mask[i] and np.all(settled_mask[i:]):
                    return t[i]

            return self.t_span[1]  # Didn't stay settled

        elif self.objective_type == "critical_damping":
            # Critical damping: c = 2 * sqrt(k * m)
            critical_c = 2.0 * np.sqrt(k * self.mass)
            damping_ratio = c / critical_c

            # Minimize deviation from critical damping (ratio = 1)
            # Also penalize slow settling
            deviation = abs(damping_ratio - 1.0)

            # Check for overshoot (indicates underdamping)
            overshoot = max(0, np.max(x) - self.x0) if self.x0 > 0 else 0
            overshoot += max(0, -np.min(x)) if self.x0 > 0 else 0

            return deviation + 0.1 * overshoot

        elif self.objective_type == "overshoot":
            # Minimize maximum overshoot
            if self.x0 > 0:
                # For positive initial displacement moving toward 0
                overshoot = max(0, -np.min(x))  # Crossing below zero
            else:
                overshoot = max(0, np.max(x))  # Crossing above zero

            return overshoot

        else:
            raise ValueError(f"Unknown objective_type: {self.objective_type}")
