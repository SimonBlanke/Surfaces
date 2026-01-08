# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2011 engineering optimization problems P09-P11.

These problems are implemented from scratch based on the mathematical
formulations in the CEC 2011 technical report and standard references.

References
----------
Das, S. & Suganthan, P. N. (2010). Problem Definitions and Evaluation
Criteria for CEC 2011 Competition on Testing Evolutionary Algorithms
on Real World Optimization Problems. Technical Report.

Soroudi, A. (2017). Power System Optimization Modeling in GAMS. Springer.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ._base_cec2011 import CEC2011Function


# =============================================================================
# P09: Circular Antenna Array Design
# =============================================================================


class CircularAntennaArray(CEC2011Function):
    """P09: Circular Antenna Array Design.

    Optimize the excitation amplitudes of a uniform circular antenna array
    to minimize the peak sidelobe level (PSL) while maintaining a desired
    main beam direction.

    The array factor for a uniform circular array is:
        AF(phi) = sum_{n=0}^{N-1} I_n * exp(j * k * r * cos(phi - phi_n))

    where:
        - N = number of elements
        - I_n = excitation amplitude of element n (decision variable)
        - k = 2*pi/lambda (wave number)
        - r = array radius
        - phi_n = 2*pi*n/N (angular position of element n)
        - phi = observation angle

    The objective is to minimize the peak sidelobe level (PSL):
        PSL = max_{phi in sidelobe region} |AF(phi)| / max |AF(phi)|

    Dimension: 12 (excitation amplitudes for 12 elements)
    Bounds: [0.1, 2.0]
    Global optimum: PSL around -11 to -15 dB for optimized arrays

    References
    ----------
    Goto, N. & Tsunoda, Y. (1977). Sidelobe reduction of circular arrays
    with a constant excitation amplitude. IEEE Trans. Antennas Propag.
    """

    _fixed_dim = 12
    _problem_id = 9

    _spec = {
        "problem_id": 9,
        "default_bounds": (0.1, 2.0),
        "continuous": True,
        "differentiable": False,  # Due to max operation
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    def __init__(
        self,
        n_elements: int = 12,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        self.n_elements = n_elements
        self._fixed_dim = n_elements
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

        # Array geometry parameters
        self._lambda = 1.0  # Normalized wavelength
        self._k = 2.0 * np.pi / self._lambda  # Wave number
        # Radius for approximately half-wavelength spacing
        self._radius = 0.5 * self._lambda * n_elements / (2.0 * np.pi)
        # Element angular positions
        self._phi_n = 2.0 * np.pi * np.arange(n_elements) / n_elements

        # Pre-compute observation angles (360 points, 1 degree resolution)
        self._phi_obs = np.linspace(0, 2 * np.pi, 360, endpoint=False)

        # Main beam direction (broadside, phi = 0)
        self._main_beam_width = 2.0 * np.pi / n_elements  # Approximate beamwidth

        # Best known value (good PSL for 12-element array is around 0.2-0.3 linear)
        self._f_global = 0.2
        self._x_global = None

    def _compute_array_factor(self, amplitudes: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Compute array factor magnitude at given angles.

        Parameters
        ----------
        amplitudes : np.ndarray
            Excitation amplitudes, shape (N,)
        phi : np.ndarray
            Observation angles, shape (M,)

        Returns
        -------
        np.ndarray
            Array factor magnitudes, shape (M,)
        """
        N = len(amplitudes)
        M = len(phi)

        # Compute array factor: AF = sum_n I_n * exp(j * k * r * cos(phi - phi_n))
        AF = np.zeros(M, dtype=complex)
        for n in range(N):
            phase = self._k * self._radius * np.cos(phi - self._phi_n[n])
            AF += amplitudes[n] * np.exp(1j * phase)

        return np.abs(AF)

    def _compute_psl(self, amplitudes: np.ndarray) -> float:
        """Compute peak sidelobe level.

        Returns the ratio of maximum sidelobe to main beam peak.
        """
        AF_mag = self._compute_array_factor(amplitudes, self._phi_obs)

        # Find main beam peak
        main_peak_idx = np.argmax(AF_mag)
        main_peak_val = AF_mag[main_peak_idx]

        if main_peak_val < 1e-10:
            return 1.0  # Degenerate case

        # Find sidelobes (outside main beam region)
        # Main beam is approximately +/- beamwidth/2 from peak
        beamwidth_indices = int(self._main_beam_width / (2 * np.pi) * len(self._phi_obs))
        beamwidth_indices = max(beamwidth_indices, 5)  # At least 5 degrees

        # Create mask for sidelobe region
        sidelobe_mask = np.ones(len(self._phi_obs), dtype=bool)
        for offset in range(-beamwidth_indices, beamwidth_indices + 1):
            idx = (main_peak_idx + offset) % len(self._phi_obs)
            sidelobe_mask[idx] = False

        if not np.any(sidelobe_mask):
            return 0.0  # No sidelobes (unlikely)

        # Peak sidelobe level (linear, not dB)
        max_sidelobe = np.max(AF_mag[sidelobe_mask])
        psl = max_sidelobe / main_peak_val

        return psl

    def _create_objective_function(self) -> None:
        """Create the circular antenna array objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            amplitudes = self._params_to_array(params)
            return self._compute_psl(amplitudes)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Batch evaluation (sequential due to array factor computation)."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        results = xp.zeros(n_points, dtype=X.dtype)

        for i in range(n_points):
            results[i] = self._compute_psl(np.asarray(X[i]))

        return results


# =============================================================================
# P10: Dynamic Economic Dispatch
# =============================================================================


class DynamicEconomicDispatch(CEC2011Function):
    """P10: Dynamic Economic Dispatch with Valve-Point Effect.

    Schedule power generation from multiple thermal units over 24 hours
    to minimize total fuel cost while meeting load demand and satisfying
    operational constraints.

    Cost function with valve-point effect:
        C_i(P_i) = a_i*P_i^2 + b_i*P_i + c_i + |d_i*sin(e_i*(Pmin_i - P_i))|

    Constraints:
        - Power balance: sum_i P_i(t) = Demand(t) + Losses(t)
        - Capacity limits: Pmin_i <= P_i(t) <= Pmax_i
        - Ramp rate limits: |P_i(t) - P_i(t-1)| <= RampRate_i

    Dimension: 5 * 24 = 120 (5 generators, 24 hours)
    Bounds: Generator-specific [Pmin, Pmax]
    Global optimum: Approximately 40,000-45,000 $/day

    References
    ----------
    Soroudi, A. (2017). Power System Optimization Modeling in GAMS. Springer.
    """

    _fixed_dim = 120  # 5 generators * 24 hours
    _problem_id = 10

    _spec = {
        "problem_id": 10,
        "default_bounds": (0.0, 300.0),  # Approximate, actual bounds vary by generator
        "continuous": True,
        "differentiable": False,  # Due to absolute value in valve-point term
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    # Generator data: [a, b, c, d, e, Pmin, Pmax, RampUp, RampDown]
    # Extended to 5 units based on standard test cases
    _generator_data = np.array(
        [
            [0.00028, 8.10, 550, 300, 0.035, 10, 125, 30, 30],  # Unit 1
            [0.00056, 8.10, 309, 200, 0.042, 20, 150, 30, 30],  # Unit 2
            [0.00056, 8.10, 307, 200, 0.042, 30, 175, 40, 40],  # Unit 3
            [0.00324, 7.74, 240, 150, 0.063, 40, 250, 50, 50],  # Unit 4
            [0.00324, 7.74, 240, 150, 0.063, 50, 300, 50, 50],  # Unit 5
        ]
    )

    # 24-hour load demand profile (MW)
    _load_demand = np.array(
        [
            410,
            430,
            450,
            470,
            490,
            520,
            550,
            580,
            620,
            650,
            670,
            700,
            690,
            660,
            640,
            620,
            600,
            590,
            570,
            540,
            510,
            480,
            450,
            420,
        ]
    )

    def __init__(
        self,
        n_generators: int = 5,
        n_hours: int = 24,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        self.n_generators = n_generators
        self.n_hours = n_hours
        self._fixed_dim = n_generators * n_hours
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

        # Penalty coefficient for constraint violations
        self._penalty_coef = 1000.0

        # Best known value (approximate)
        self._f_global = 40000.0
        self._x_global = None

    def _compute_cost(self, P: np.ndarray) -> float:
        """Compute total fuel cost over 24 hours.

        Parameters
        ----------
        P : np.ndarray
            Power output, shape (n_generators * n_hours,)
            Arranged as [P1_h1, P2_h1, ..., P5_h1, P1_h2, ..., P5_h24]

        Returns
        -------
        float
            Total fuel cost with penalty for constraint violations.
        """
        # Reshape to (n_hours, n_generators)
        P_matrix = P.reshape(self.n_hours, self.n_generators)

        total_cost = 0.0
        penalty = 0.0

        for t in range(self.n_hours):
            hour_cost = 0.0
            total_power = 0.0

            for i in range(self.n_generators):
                P_i = P_matrix[t, i]
                a, b, c, d, e, Pmin, Pmax, RU, RD = self._generator_data[i]

                # Capacity constraint penalty
                if P_i < Pmin:
                    penalty += self._penalty_coef * (Pmin - P_i) ** 2
                    P_i = Pmin
                elif P_i > Pmax:
                    penalty += self._penalty_coef * (P_i - Pmax) ** 2
                    P_i = Pmax

                # Cost with valve-point effect
                fuel_cost = a * P_i**2 + b * P_i + c
                valve_point = abs(d * np.sin(e * (Pmin - P_i)))
                hour_cost += fuel_cost + valve_point

                total_power += P_i

                # Ramp rate constraint (for t > 0)
                if t > 0:
                    P_prev = P_matrix[t - 1, i]
                    ramp = abs(P_i - P_prev)
                    if ramp > RU:
                        penalty += self._penalty_coef * (ramp - RU) ** 2

            total_cost += hour_cost

            # Power balance constraint
            demand = self._load_demand[t]
            power_imbalance = abs(total_power - demand)
            if power_imbalance > 1.0:  # 1 MW tolerance
                penalty += self._penalty_coef * power_imbalance**2

        return total_cost + penalty

    def _create_objective_function(self) -> None:
        """Create the dynamic economic dispatch objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            P = self._params_to_array(params)
            return self._compute_cost(P)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        results = xp.zeros(n_points, dtype=X.dtype)

        for i in range(n_points):
            results[i] = self._compute_cost(np.asarray(X[i]))

        return results


# =============================================================================
# P11: Hydrothermal Scheduling
# =============================================================================


class HydrothermalScheduling(CEC2011Function):
    """P11: Short-Term Hydrothermal Scheduling.

    Schedule power generation from thermal and hydro units over 24 hours
    to minimize total thermal fuel cost while meeting load demand and
    satisfying hydraulic constraints.

    The hydro power is a function of water discharge and reservoir head:
        P_hydro = C1 * V^2 + C2 * q^2 + C3 * V * q + C4 * V + C5 * q + C6

    Water balance constraint:
        V(t+1) = V(t) + Inflow(t) - q(t) - Spillage(t)

    Dimension: 24 (hydro discharge for each hour, thermal computed to meet demand)
    Bounds: [5, 15] for discharge rate (10^4 acre-ft)
    Global optimum: Depends on specific test case

    References
    ----------
    Wood, A.J. & Wollenberg, B.F. (1996). Power Generation, Operation
    and Control. Wiley.
    """

    _fixed_dim = 24  # Hydro discharge for 24 hours
    _problem_id = 11

    _spec = {
        "problem_id": 11,
        "default_bounds": (5.0, 15.0),  # Discharge rate bounds
        "continuous": True,
        "differentiable": True,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    # Thermal generator cost coefficients [a, b, c, Pmin, Pmax]
    _thermal_data = np.array(
        [
            [0.0050, 2.00, 0, 0, 500],  # Thermal unit 1 (baseload)
        ]
    )

    # Hydro power coefficients: P = C1*V^2 + C2*q^2 + C3*V*q + C4*V + C5*q + C6
    # Simplified model where power depends mainly on discharge
    _hydro_coeffs = np.array([-0.0042, -0.42, 0.030, 0.90, 10.0, -50.0])

    # Reservoir parameters
    _V_initial = 100.0  # Initial volume (10^4 acre-ft)
    _V_final = 100.0  # Required final volume
    _V_min = 60.0  # Minimum volume
    _V_max = 120.0  # Maximum volume
    _inflow = 2.5  # Constant inflow per hour (10^4 acre-ft)

    # 24-hour load demand (MW)
    _load_demand = np.array(
        [
            400,
            390,
            380,
            370,
            370,
            380,
            520,
            620,
            740,
            840,
            900,
            920,
            900,
            860,
            800,
            740,
            700,
            700,
            720,
            680,
            620,
            540,
            460,
            410,
        ]
    )

    def __init__(
        self,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

        self._penalty_coef = 10000.0

        # Best known value (approximate)
        self._f_global = 900000.0
        self._x_global = None

    def _compute_hydro_power(self, V: float, q: float) -> float:
        """Compute hydro power output from volume and discharge.

        Parameters
        ----------
        V : float
            Reservoir volume (10^4 acre-ft)
        q : float
            Water discharge rate (10^4 acre-ft/hour)

        Returns
        -------
        float
            Hydro power output (MW)
        """
        C = self._hydro_coeffs
        P_hydro = C[0] * V**2 + C[1] * q**2 + C[2] * V * q + C[3] * V + C[4] * q + C[5]
        return max(0.0, P_hydro)

    def _compute_cost(self, q: np.ndarray) -> float:
        """Compute total thermal cost for given hydro discharge schedule.

        Parameters
        ----------
        q : np.ndarray
            Hydro discharge for each hour, shape (24,)

        Returns
        -------
        float
            Total thermal fuel cost with penalty for violations.
        """
        total_cost = 0.0
        penalty = 0.0

        V = self._V_initial  # Start with initial volume

        for t in range(self.n_dim):
            q_t = q[t]

            # Water balance: V(t+1) = V(t) + inflow - discharge
            V_next = V + self._inflow - q_t

            # Reservoir volume constraints
            if V_next < self._V_min:
                penalty += self._penalty_coef * (self._V_min - V_next) ** 2
            elif V_next > self._V_max:
                penalty += self._penalty_coef * (V_next - self._V_max) ** 2

            # Compute hydro power
            P_hydro = self._compute_hydro_power(V, q_t)

            # Thermal power to meet demand
            demand = self._load_demand[t]
            P_thermal = demand - P_hydro

            # Thermal capacity constraints
            a, b, c, Pmin, Pmax = self._thermal_data[0]
            if P_thermal < Pmin:
                penalty += self._penalty_coef * (Pmin - P_thermal) ** 2
                P_thermal = Pmin
            elif P_thermal > Pmax:
                penalty += self._penalty_coef * (P_thermal - Pmax) ** 2
                P_thermal = Pmax

            # Thermal cost (quadratic)
            thermal_cost = a * P_thermal**2 + b * P_thermal + c
            total_cost += thermal_cost

            # Update volume for next hour
            V = max(self._V_min, min(self._V_max, V_next))

        # Final volume constraint
        final_volume_error = abs(V - self._V_final)
        if final_volume_error > 1.0:
            penalty += self._penalty_coef * final_volume_error**2

        return total_cost + penalty

    def _create_objective_function(self) -> None:
        """Create the hydrothermal scheduling objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            q = self._params_to_array(params)
            return self._compute_cost(q)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        results = xp.zeros(n_points, dtype=X.dtype)

        for i in range(n_points):
            results[i] = self._compute_cost(np.asarray(X[i]))

        return results


# =============================================================================
# Collections
# =============================================================================

CEC2011_ENGINEERING: List[type] = [
    CircularAntennaArray,
    DynamicEconomicDispatch,
    HydrothermalScheduling,
]
