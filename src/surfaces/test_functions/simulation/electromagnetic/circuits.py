# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Electrical circuit test functions based on ODE systems."""

from typing import Any, Dict

import numpy as np

from ..dynamics._base_ode import ODESimulationFunction


class RLCCircuitFunction(ODESimulationFunction):
    """RLC series circuit optimization problem.

    Models a series RLC circuit with voltage source V(t):

        L * dI/dt + R * I + (1/C) * Q = V(t)
        dQ/dt = I

    where I is current, Q is charge, and V(t) is the input voltage.

    The optimization objective is to find R, L, C values that achieve
    desired frequency response or transient behavior.

    Parameters
    ----------
    V_amplitude : float, default=1.0
        Amplitude of input voltage.
    V_frequency : float, default=1.0
        Frequency of input voltage (Hz). Set to 0 for step input.
    t_span : tuple, default=(0, 10)
        Time interval for simulation.
    target_frequency : float, optional
        Target resonant frequency (Hz). Used for "resonance" objective.
    objective_type : str, default="resonance"
        Type of objective:
        - "resonance": Match target resonant frequency
        - "damping": Achieve critical damping
        - "bandwidth": Maximize bandwidth around resonance
    **kwargs
        Additional arguments passed to ODESimulationFunction.

    Search Space
    ------------
    R : float in [0.1, 100.0]
        Resistance (Ohms).
    L : float in [0.001, 1.0]
        Inductance (Henries).
    C : float in [1e-6, 1e-3]
        Capacitance (Farads).

    Notes
    -----
    Natural frequency: omega_0 = 1 / sqrt(L * C)
    Resonant frequency: f_0 = omega_0 / (2 * pi)
    Quality factor: Q = (1/R) * sqrt(L/C)
    Critical damping: R = 2 * sqrt(L/C)

    Examples
    --------
    >>> func = RLCCircuitFunction(target_frequency=100.0)
    >>> result = func({"R": 10.0, "L": 0.01, "C": 0.0001})

    References
    ----------
    .. [1] Horowitz, P. & Hill, W. (2015). The Art of Electronics, 3rd ed.
    """

    _spec = {
        "simulation_based": True,
        "expensive": False,
        "continuous": True,
        "ode_based": True,
        "n_dim": 3,
        "unimodal": False,  # Multiple local optima in some objectives
        "domain": "electronics",
    }

    def __init__(
        self,
        V_amplitude: float = 1.0,
        V_frequency: float = 1.0,
        t_span: tuple = (0, 10),
        target_frequency: float = 10.0,
        objective_type: str = "resonance",
        **kwargs,
    ) -> None:
        self.V_amplitude = V_amplitude
        self.V_frequency = V_frequency
        self.target_frequency = target_frequency
        self.objective_type = objective_type
        # High resolution for frequency analysis
        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        super().__init__(t_span=t_span, t_eval=t_eval, **kwargs)

    @property
    def search_space(self) -> Dict[str, np.ndarray]:
        """3D search space for circuit parameters."""
        return {
            "R": np.linspace(0.1, 100.0, 100),
            "L": np.linspace(0.001, 1.0, 100),
            "C": np.linspace(1e-6, 1e-3, 100),
        }

    def _get_initial_conditions(self) -> np.ndarray:
        """Return initial conditions [Q (charge), I (current)]."""
        return np.array([0.0, 0.0])

    def _voltage_input(self, t: float) -> float:
        """Generate input voltage signal."""
        if self.V_frequency == 0:
            # Step input
            return self.V_amplitude
        else:
            # Sinusoidal input
            return self.V_amplitude * np.sin(2 * np.pi * self.V_frequency * t)

    def _ode_system(self, t: float, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """RLC circuit ODE system.

        dQ/dt = I
        dI/dt = (V(t) - R*I - Q/C) / L
        """
        Q, current = y
        R = params["R"]
        L = params["L"]
        C = params["C"]

        V = self._voltage_input(t)

        dQ_dt = current
        dI_dt = (V - R * current - Q / C) / L

        return np.array([dQ_dt, dI_dt])

    def _compute_objective(self, t: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        """Compute objective from circuit response."""
        # y[0, :] = Charge, y[1, :] = Current (extracted in subclasses if needed)

        R = params["R"]
        L = params["L"]
        C = params["C"]

        # Check for numerical issues
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return float("inf")

        if self.objective_type == "resonance":
            # Match target resonant frequency
            # Natural frequency: f_0 = 1 / (2*pi*sqrt(L*C))
            actual_freq = 1.0 / (2 * np.pi * np.sqrt(L * C))
            freq_error = abs(actual_freq - self.target_frequency)

            # Also penalize high damping (want good resonance)
            Q_factor = (1 / R) * np.sqrt(L / C)
            damping_penalty = max(0, 1.0 - Q_factor)  # Penalize Q < 1

            return freq_error + 0.1 * damping_penalty

        elif self.objective_type == "damping":
            # Achieve critical damping: R = 2*sqrt(L/C)
            critical_R = 2 * np.sqrt(L / C)
            damping_ratio = R / critical_R

            # Want damping_ratio = 1 (critical damping)
            return abs(damping_ratio - 1.0)

        elif self.objective_type == "bandwidth":
            # Maximize bandwidth (lower Q factor means wider bandwidth)
            # But not too low (need some resonance)
            Q_factor = (1 / R) * np.sqrt(L / C)

            # Optimal Q_factor around 0.5-2 for good bandwidth
            if Q_factor < 0.1:
                return float("inf")  # Too overdamped

            # Bandwidth = f_0 / Q
            f_0 = 1.0 / (2 * np.pi * np.sqrt(L * C))
            bandwidth = f_0 / Q_factor

            # Maximize bandwidth (minimize negative)
            return -bandwidth

        else:
            raise ValueError(f"Unknown objective_type: {self.objective_type}")


class RCFilterFunction(ODESimulationFunction):
    """RC low-pass filter optimization problem.

    Models a simple RC low-pass filter:

        dV_out/dt = (V_in - V_out) / (R * C)

    The optimization objective is to find R and C values that achieve
    a target cutoff frequency with minimal ripple.

    Parameters
    ----------
    V_in_dc : float, default=1.0
        DC component of input voltage.
    V_in_ac : float, default=0.5
        AC component amplitude of input voltage.
    input_frequency : float, default=100.0
        Frequency of AC component (Hz).
    t_span : tuple, default=(0, 0.1)
        Time interval for simulation.
    target_cutoff : float, default=50.0
        Target cutoff frequency (Hz).
    objective_type : str, default="cutoff"
        Type of objective:
        - "cutoff": Match target cutoff frequency
        - "ripple": Minimize output ripple at input frequency
        - "settling": Minimize settling time for step response
    **kwargs
        Additional arguments passed to ODESimulationFunction.

    Search Space
    ------------
    R : float in [100, 100000]
        Resistance (Ohms).
    C : float in [1e-9, 1e-5]
        Capacitance (Farads).

    Notes
    -----
    Cutoff frequency: f_c = 1 / (2 * pi * R * C)
    Time constant: tau = R * C

    Examples
    --------
    >>> func = RCFilterFunction(target_cutoff=1000.0)
    >>> result = func({"R": 1000.0, "C": 1.59e-7})  # ~1kHz cutoff

    References
    ----------
    .. [1] Sedra, A.S. & Smith, K.C. (2014). Microelectronic Circuits, 7th ed.
    """

    _spec = {
        "simulation_based": True,
        "expensive": False,
        "continuous": True,
        "ode_based": True,
        "n_dim": 2,
        "unimodal": True,
        "domain": "electronics",
    }

    def __init__(
        self,
        V_in_dc: float = 1.0,
        V_in_ac: float = 0.5,
        input_frequency: float = 100.0,
        t_span: tuple = (0, 0.1),
        target_cutoff: float = 50.0,
        objective_type: str = "cutoff",
        **kwargs,
    ) -> None:
        self.V_in_dc = V_in_dc
        self.V_in_ac = V_in_ac
        self.input_frequency = input_frequency
        self.target_cutoff = target_cutoff
        self.objective_type = objective_type
        # High resolution for ripple measurement
        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        super().__init__(t_span=t_span, t_eval=t_eval, **kwargs)

    @property
    def search_space(self) -> Dict[str, np.ndarray]:
        """2D search space for filter parameters."""
        return {
            "R": np.logspace(2, 5, 100),  # 100 to 100k Ohms
            "C": np.logspace(-9, -5, 100),  # 1nF to 10uF
        }

    def _get_initial_conditions(self) -> np.ndarray:
        """Return initial output voltage."""
        return np.array([0.0])

    def _voltage_input(self, t: float) -> float:
        """Generate input voltage signal."""
        return self.V_in_dc + self.V_in_ac * np.sin(2 * np.pi * self.input_frequency * t)

    def _ode_system(self, t: float, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """RC filter ODE system.

        dV_out/dt = (V_in - V_out) / (R * C)
        """
        V_out = y[0]
        R = params["R"]
        C = params["C"]

        V_in = self._voltage_input(t)
        tau = R * C

        dV_out_dt = (V_in - V_out) / tau

        return np.array([dV_out_dt])

    def _compute_objective(self, t: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        """Compute objective from filter response."""
        V_out = y[0, :]
        R = params["R"]
        C = params["C"]

        # Check for numerical issues
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return float("inf")

        if self.objective_type == "cutoff":
            # Match target cutoff frequency
            # Cutoff: f_c = 1 / (2*pi*R*C)
            actual_cutoff = 1.0 / (2 * np.pi * R * C)

            # Use relative error for better scaling
            relative_error = abs(actual_cutoff - self.target_cutoff) / self.target_cutoff
            return relative_error

        elif self.objective_type == "ripple":
            # Minimize output ripple (peak-to-peak variation in steady state)
            # Use last 50% of signal (after transient)
            steady_state = V_out[len(V_out) // 2 :]
            ripple = np.max(steady_state) - np.min(steady_state)

            # Also penalize deviation from DC level
            dc_error = abs(np.mean(steady_state) - self.V_in_dc)

            return ripple + 0.1 * dc_error

        elif self.objective_type == "settling":
            # Minimize settling time (for step response, V_in_ac=0)
            # Time to reach 98% of final value
            target = self.V_in_dc
            threshold = 0.98 * target

            settled_mask = V_out >= threshold
            if not np.any(settled_mask):
                return self.t_span[1] * 2  # Penalty

            settling_idx = np.argmax(settled_mask)
            return t[settling_idx]

        else:
            raise ValueError(f"Unknown objective_type: {self.objective_type}")
