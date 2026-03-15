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
        - "resonance": Match target resonant frequency (from FFT of step response)
        - "damping": Achieve critical damping (from transient overshoot and settling)
        - "bandwidth": Maximize -3dB bandwidth (from FFT of step response)
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
        "eval_cost": 3122.7,
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

    def _default_search_space(self) -> Dict[str, np.ndarray]:
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
        """Generate input voltage signal.

        Built-in objectives use step input for transient and spectral
        analysis of the circuit's natural response.
        """
        if self.V_frequency == 0 or self.objective_type in ("resonance", "damping", "bandwidth"):
            return self.V_amplitude
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
        """Compute objective from circuit step response.

        All objectives analyze the actual simulation waveform rather than
        using closed-form formulas.
        """
        current = y[1, :]

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return float("inf")

        if self.objective_type == "resonance":
            # Measure resonant frequency from FFT of step response current
            dt = t[1] - t[0]
            window = np.hanning(len(current))
            spectrum = np.abs(np.fft.rfft(current * window))
            freqs = np.fft.rfftfreq(len(current), d=dt)
            spectrum[0] = 0  # skip DC

            if np.max(spectrum) < 1e-12:
                return float("inf")

            peak_idx = np.argmax(spectrum)
            measured_freq = freqs[peak_idx]

            return abs(measured_freq - self.target_frequency)

        elif self.objective_type == "damping":
            # Measure critical damping from actual transient behavior:
            # no overshoot (current stays positive) + fastest settling
            peak_val = np.max(np.abs(current))
            if peak_val < 1e-12:
                return float("inf")

            # Overshoot: for step input, current should rise then decay
            # to zero without going negative (critically damped)
            peak_idx = np.argmax(current)
            post_peak_min = np.min(current[peak_idx:])
            overshoot = max(0, -post_peak_min) / peak_val

            # Settling time (2% of peak value)
            threshold = 0.02 * peak_val
            settled_mask = np.abs(current) < threshold
            settling_time = t[-1]
            for i in range(len(settled_mask)):
                if settled_mask[i] and np.all(settled_mask[i:]):
                    settling_time = t[i]
                    break

            norm_settling = settling_time / t[-1]

            return overshoot + norm_settling

        elif self.objective_type == "bandwidth":
            # Measure -3dB bandwidth from FFT of step response current
            dt = t[1] - t[0]
            window = np.hanning(len(current))
            spectrum = np.abs(np.fft.rfft(current * window))
            freqs = np.fft.rfftfreq(len(current), d=dt)
            spectrum[0] = 0  # skip DC

            peak_mag = np.max(spectrum)
            if peak_mag < 1e-12:
                return float("inf")

            peak_idx = np.argmax(spectrum)
            threshold_3db = peak_mag / np.sqrt(2)
            above_3db = spectrum >= threshold_3db

            # Walk outward from peak to find -3dB edges
            left = peak_idx
            while left > 0 and above_3db[left - 1]:
                left -= 1

            right = peak_idx
            while right < len(above_3db) - 1 and above_3db[right + 1]:
                right += 1

            bandwidth = freqs[right] - freqs[left]

            if bandwidth < 1e-12:
                return float("inf")

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
        - "cutoff": Match target cutoff frequency (from measured output gain)
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
        "eval_cost": 7542.7,
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
        self.target_cutoff = target_cutoff
        self.objective_type = objective_type
        # For cutoff measurement, drive at target frequency to test -3dB gain
        self.input_frequency = target_cutoff if objective_type == "cutoff" else input_frequency
        # High resolution for ripple measurement
        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        super().__init__(t_span=t_span, t_eval=t_eval, **kwargs)

    def _default_search_space(self) -> Dict[str, np.ndarray]:
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

        # Check for numerical issues
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return float("inf")

        if self.objective_type == "cutoff":
            # Measure actual gain from simulation output in steady state.
            # The simulation drives at the target cutoff frequency, so
            # the gain should be 1/sqrt(2) (-3dB) when R,C match.
            n = len(V_out)
            steady_out = V_out[n // 2 :]
            out_ac = (np.max(steady_out) - np.min(steady_out)) / 2

            if self.V_in_ac < 1e-12:
                return float("inf")

            gain = out_ac / self.V_in_ac
            target_gain = 1.0 / np.sqrt(2.0)
            return abs(gain - target_gain)

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
