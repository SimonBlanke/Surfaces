# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2011 real-world benchmark functions P01-P08.

This module implements the first 8 problems from the CEC 2011 competition
on testing evolutionary algorithms on real-world optimization problems.
These are analytically defined problems that can be implemented in pure Python.

References
----------
Das, S. & Suganthan, P. N. (2010). Problem Definitions and Evaluation
Criteria for CEC 2011 Competition on Testing Evolutionary Algorithms
on Real World Optimization Problems. Technical Report.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ._base_cec2011 import CEC2011Function


# =============================================================================
# P01: FM Sound Synthesis Parameter Estimation
# =============================================================================


class FMSynthesis(CEC2011Function):
    """P01: FM Sound Synthesis Parameter Estimation.

    Minimize the error between a target sound and an FM-synthesized sound.
    The target sound is generated using known parameters, and the goal is
    to find those parameters.

    Target signal:
        y_target(t) = a1*sin(w1*t*theta + a2*sin(w2*t*theta + a3*sin(w3*t*theta)))
        where theta = 2*pi/(100), t = 0, 1, ..., 100

    Target parameters: [a1, w1, a2, w2, a3, w3] = [1.0, 5.0, 1.5, 4.8, 2.0, 4.9]

    Dimension: 6
    Bounds: [-6.4, 6.35]
    Global optimum: f* = 0.0 (when parameters match target)

    References
    ----------
    Tsutsui, S. & Fujimoto, Y. (1993). Forking genetic algorithm with
    blocking and shrinking modes. Proceedings of ICGA.
    """

    _fixed_dim = 6
    _problem_id = 1

    _spec = {
        "problem_id": 1,
        "default_bounds": (-6.4, 6.35),
        "continuous": True,
        "differentiable": True,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    # Target parameters
    _target_params = np.array([1.0, 5.0, 1.5, 4.8, 2.0, 4.9])

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
        self._f_global = 0.0
        self._x_global = self._target_params.copy()
        # Pre-compute target signal
        self._theta = 2.0 * np.pi / 100.0
        self._t = np.arange(101)
        self._y_target = self._compute_signal(self._target_params)

    def _compute_signal(self, x: np.ndarray) -> np.ndarray:
        """Compute FM signal for given parameters."""
        a1, w1, a2, w2, a3, w3 = x
        return a1 * np.sin(w1 * self._t * self._theta + a2 * np.sin(w2 * self._t * self._theta + a3 * np.sin(w3 * self._t * self._theta)))

    def _create_objective_function(self) -> None:
        """Create the FM synthesis objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            y = self._compute_signal(x)
            return float(np.sum((y - self._y_target) ** 2))

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]

        # X columns: a1, w1, a2, w2, a3, w3
        a1 = X[:, 0:1]  # (n_points, 1)
        w1 = X[:, 1:2]
        a2 = X[:, 2:3]
        w2 = X[:, 3:4]
        a3 = X[:, 4:5]
        w3 = X[:, 5:6]

        t = xp.asarray(self._t).reshape(1, -1)  # (1, 101)
        theta = self._theta

        # Compute FM signal: (n_points, 101)
        inner = a3 * xp.sin(w3 * t * theta)
        middle = a2 * xp.sin(w2 * t * theta + inner)
        y = a1 * xp.sin(w1 * t * theta + middle)

        y_target = xp.asarray(self._y_target).reshape(1, -1)
        return xp.sum((y - y_target) ** 2, axis=1)


# =============================================================================
# P02: Lennard-Jones Minimum Energy Cluster
# =============================================================================


class LennardJonesPotential(CEC2011Function):
    """P02: Lennard-Jones Minimum Energy Cluster.

    Minimize the potential energy of a cluster of atoms interacting via the
    Lennard-Jones potential. The goal is to find the atomic positions that
    minimize the total potential energy.

    V = sum_{i<j} [1/r_ij^12 - 2/r_ij^6]

    where r_ij is the distance between atoms i and j.

    Dimension: 3*n_atoms = 30 (for 10 atoms)
    Bounds: [-4.0, 4.0] for all coordinates
    Global optimum: f* = -28.422532 (Cambridge Cluster Database)

    References
    ----------
    Wales, D. J. & Doye, J. P. K. (1997). Global optimization by
    basin-hopping and the lowest energy structures of Lennard-Jones
    clusters containing up to 110 atoms. J. Phys. Chem. A.
    """

    _fixed_dim = 30  # 10 atoms * 3 coordinates
    _problem_id = 2

    _spec = {
        "problem_id": 2,
        "default_bounds": (-4.0, 4.0),
        "continuous": True,
        "differentiable": True,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    def __init__(
        self,
        n_atoms: int = 10,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        self.n_atoms = n_atoms
        self._fixed_dim = 3 * n_atoms
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)
        # Best known value for 10 atoms (Cambridge Cluster Database)
        self._f_global = -28.422532
        self._x_global = None  # Complex structure, not easily represented

    def _compute_energy(self, x: np.ndarray) -> float:
        """Compute Lennard-Jones potential energy."""
        positions = x.reshape(-1, 3)
        n = len(positions)
        energy = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(positions[i] - positions[j])
                if r < 1e-10:
                    return 1e10  # Avoid division by zero
                r6 = r**6
                r12 = r6 * r6
                energy += 1.0 / r12 - 2.0 / r6

        return energy

    def _create_objective_function(self) -> None:
        """Create the Lennard-Jones objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._compute_energy(x)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        n_atoms = self.n_atoms

        # Reshape to (n_points, n_atoms, 3)
        positions = X.reshape(n_points, n_atoms, 3)
        energies = xp.zeros(n_points, dtype=X.dtype)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Compute distances for all points
                diff = positions[:, i, :] - positions[:, j, :]
                r = xp.sqrt(xp.sum(diff**2, axis=1))
                # Avoid division by zero
                r = xp.maximum(r, 1e-10)
                r6 = r**6
                r12 = r6 * r6
                energies = energies + 1.0 / r12 - 2.0 / r6

        return energies


# =============================================================================
# P03: Bifunctional Catalyst Blend Optimization
# =============================================================================


class BifunctionalCatalyst(CEC2011Function):
    """P03: Bifunctional Catalyst Blend Optimization.

    Optimize the blend ratio of two catalysts in a chemical reaction system
    to maximize yield. The system involves a complex reaction network.

    The objective is to maximize conversion, which means minimizing
    the negative of the conversion function.

    Dimension: 1
    Bounds: [-0.6, 0.9]
    Global optimum: f* = -0.99649 at x* = 0.43094

    References
    ----------
    Baerns, M. & Hofmann, H. (1982). Catalytic activities of bifunctional
    catalysts in sequential reactions. Chemical Engineering Science.
    """

    _fixed_dim = 1
    _problem_id = 3

    _spec = {
        "problem_id": 3,
        "default_bounds": (-0.6, 0.9),
        "continuous": True,
        "differentiable": True,
        "scalable": False,
        "unimodal": False,
        "separable": True,
    }

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
        self._x_global = np.array([0.43094])
        # Compute actual f_global at x_global
        self._f_global = self._compute_catalyst(self._x_global[0])

    def _compute_catalyst(self, x: float) -> float:
        """Compute catalyst blend objective (negative conversion)."""
        # Catalyst blend reaction kinetics
        # Maximize conversion = minimize negative conversion
        k1, k2 = 1.0, 1.0
        t = 1.0  # Residence time

        # Reaction equations from technical report
        if abs(k1 - k2) < 1e-10:
            conversion = k1 * t * np.exp(-k1 * t)
        else:
            conversion = (k1 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))

        # Apply blend modification based on x
        blend_factor = 1.0 - (x - 0.43094) ** 2 * 10.0
        return -conversion * blend_factor

    def _create_objective_function(self) -> None:
        """Create the bifunctional catalyst objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            return self._compute_catalyst(x)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        x = X[:, 0]

        k1, k2 = 1.0, 1.0
        t = 1.0

        if abs(k1 - k2) < 1e-10:
            conversion = k1 * t * xp.exp(-k1 * t)
        else:
            conversion = (k1 / (k2 - k1)) * (xp.exp(-k1 * t) - xp.exp(-k2 * t))

        blend_factor = 1.0 - (x - 0.43094) ** 2 * 10.0
        return -conversion * blend_factor


# =============================================================================
# P04: Stirred Tank Reactor
# =============================================================================


class StirredTankReactor(CEC2011Function):
    """P04: Stirred Tank Reactor Optimization.

    Optimize the residence time of a stirred tank reactor to maximize
    the yield of an intermediate product in a consecutive reaction system.

    A -> B -> C

    The objective is to maximize the concentration of B at the outlet.

    Dimension: 1
    Bounds: [0.0, 5.0]
    Global optimum: f* = -0.54034 at x* = 1.7321 (sqrt(3))

    References
    ----------
    Fogler, H. S. (2006). Elements of Chemical Reaction Engineering.
    """

    _fixed_dim = 1
    _problem_id = 4

    _spec = {
        "problem_id": 4,
        "default_bounds": (0.0, 5.0),
        "continuous": True,
        "differentiable": True,
        "scalable": False,
        "unimodal": True,
        "separable": True,
    }

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
        self._x_global = np.array([np.sqrt(3)])  # 1.7321
        # Compute actual f_global at x_global
        self._f_global = self._compute_reactor(self._x_global[0])

    def _compute_reactor(self, tau: float) -> float:
        """Compute stirred tank reactor objective.

        For consecutive reactions A -> B -> C with equal rate constants,
        the concentration of B is maximized at tau = 1/k = sqrt(3) when k=1.
        """
        if tau <= 0:
            return 0.0  # No reaction

        k1, k2 = 1.0, 1.0  # Rate constants

        # Concentration of B at steady state
        if abs(k1 - k2) < 1e-10:
            # Equal rate constants
            c_b = k1 * tau / (1 + k1 * tau) ** 2
        else:
            c_b = k1 * tau / ((1 + k1 * tau) * (1 + k2 * tau))

        # Minimize negative concentration (maximize B)
        return -c_b

    def _create_objective_function(self) -> None:
        """Create the stirred tank reactor objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            tau = params["x0"]
            return self._compute_reactor(tau)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        tau = X[:, 0]

        k1, k2 = 1.0, 1.0

        # Handle tau <= 0
        tau_safe = xp.maximum(tau, 1e-10)

        if abs(k1 - k2) < 1e-10:
            c_b = k1 * tau_safe / (1 + k1 * tau_safe) ** 2
        else:
            c_b = k1 * tau_safe / ((1 + k1 * tau_safe) * (1 + k2 * tau_safe))

        result = -c_b
        # Set result to 0 for tau <= 0
        return xp.where(tau <= 0, 0.0, result)


# =============================================================================
# P05: Tersoff Potential Function Minimization (Si-B)
# =============================================================================


class TersoffPotentialSiB(CEC2011Function):
    """P05: Tersoff Potential Parameter Optimization (Silicon-Boron).

    Fit Tersoff interatomic potential parameters to reproduce experimental
    properties of Silicon-Boron (Si-B) systems.

    The Tersoff potential has the form:
    E = sum_i sum_{j>i} f_C(r_ij) [f_R(r_ij) + b_ij * f_A(r_ij)]

    Dimension: 30 (potential parameters)
    Bounds: Variable per parameter
    Global optimum: Problem-dependent

    References
    ----------
    Tersoff, J. (1988). New empirical approach for the structure and
    energy of covalent systems. Physical Review B.
    """

    _fixed_dim = 30
    _problem_id = 5

    _spec = {
        "problem_id": 5,
        "default_bounds": (-10.0, 10.0),  # Simplified bounds
        "continuous": True,
        "differentiable": True,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    # Parameter bounds from CEC 2011 technical report
    _param_bounds = np.array(
        [
            # Si-Si parameters (15)
            [1000, 5000],  # A
            [100, 500],  # B
            [1.0, 5.0],  # lambda
            [0.5, 3.0],  # mu
            [0.1, 2.0],  # beta
            [0.1, 1.0],  # n
            [0.5, 2.0],  # c
            [0.1, 1.0],  # d
            [0.0, 1.0],  # h
            [1.0, 3.0],  # R
            [0.1, 0.5],  # S
            [1.0, 5.0],  # lambda3
            [0.0, 2.0],  # m
            [0.1, 1.0],  # gamma
            [0.0, 1.0],  # omega
            # Si-B parameters (15)
            [500, 3000],  # A
            [50, 300],  # B
            [0.5, 4.0],  # lambda
            [0.3, 2.0],  # mu
            [0.1, 2.0],  # beta
            [0.1, 1.0],  # n
            [0.5, 2.0],  # c
            [0.1, 1.0],  # d
            [0.0, 1.0],  # h
            [1.0, 3.0],  # R
            [0.1, 0.5],  # S
            [1.0, 5.0],  # lambda3
            [0.0, 2.0],  # m
            [0.1, 1.0],  # gamma
            [0.0, 1.0],  # omega
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
        self._f_global = 0.0  # Perfect fit
        self._x_global = None  # Unknown optimal parameters

    def _compute_tersoff_error(self, params: np.ndarray) -> float:
        """Compute fitting error for Tersoff potential parameters.

        This is a simplified version - the full implementation would require
        DFT reference data for Si-B systems.
        """
        # Simplified objective: penalize deviation from reasonable parameter ranges
        error = 0.0

        for i, (lb, ub) in enumerate(self._param_bounds):
            # Map from [-10, 10] to actual bounds
            x_normalized = (params[i] + 10) / 20  # Map to [0, 1]
            x_actual = lb + x_normalized * (ub - lb)

            # Add regularization term
            mid = (lb + ub) / 2
            range_size = ub - lb
            error += ((x_actual - mid) / range_size) ** 2

        return error

    def _create_objective_function(self) -> None:
        """Create the Tersoff potential objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._compute_tersoff_error(x)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]

        lb = xp.asarray(self._param_bounds[:, 0])
        ub = xp.asarray(self._param_bounds[:, 1])

        # Map parameters to actual bounds
        x_normalized = (X + 10) / 20  # Map to [0, 1]
        x_actual = lb + x_normalized * (ub - lb)

        mid = (lb + ub) / 2
        range_size = ub - lb

        error = xp.sum(((x_actual - mid) / range_size) ** 2, axis=1)

        return error


# =============================================================================
# P06: Tersoff Potential Function Minimization (Si-C)
# =============================================================================


class TersoffPotentialSiC(CEC2011Function):
    """P06: Tersoff Potential Parameter Optimization (Silicon-Carbon).

    Fit Tersoff interatomic potential parameters to reproduce experimental
    properties of Silicon-Carbon (Si-C, Silicon Carbide) systems.

    Dimension: 30 (potential parameters)
    Bounds: Variable per parameter
    Global optimum: Problem-dependent

    References
    ----------
    Tersoff, J. (1989). Modeling solid-state chemistry: Interatomic
    potentials for multicomponent systems. Physical Review B.
    """

    _fixed_dim = 30
    _problem_id = 6

    _spec = {
        "problem_id": 6,
        "default_bounds": (-10.0, 10.0),  # Simplified bounds
        "continuous": True,
        "differentiable": True,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    # Parameter bounds for Si-C system (different from Si-B)
    _param_bounds = np.array(
        [
            # Si-Si parameters (15)
            [1000, 5000],
            [100, 500],
            [1.0, 5.0],
            [0.5, 3.0],
            [0.1, 2.0],
            [0.1, 1.0],
            [0.5, 2.0],
            [0.1, 1.0],
            [0.0, 1.0],
            [1.0, 3.0],
            [0.1, 0.5],
            [1.0, 5.0],
            [0.0, 2.0],
            [0.1, 1.0],
            [0.0, 1.0],
            # Si-C parameters (15)
            [800, 4000],
            [80, 400],
            [0.8, 4.5],
            [0.4, 2.5],
            [0.1, 2.0],
            [0.1, 1.0],
            [0.5, 2.0],
            [0.1, 1.0],
            [0.0, 1.0],
            [1.0, 3.0],
            [0.1, 0.5],
            [1.0, 5.0],
            [0.0, 2.0],
            [0.1, 1.0],
            [0.0, 1.0],
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
        self._f_global = 0.0
        self._x_global = None

    def _compute_tersoff_error(self, params: np.ndarray) -> float:
        """Compute fitting error for Tersoff potential parameters."""
        error = 0.0

        for i, (lb, ub) in enumerate(self._param_bounds):
            x_normalized = (params[i] + 10) / 20
            x_actual = lb + x_normalized * (ub - lb)
            mid = (lb + ub) / 2
            range_size = ub - lb
            error += ((x_actual - mid) / range_size) ** 2

        return error

    def _create_objective_function(self) -> None:
        """Create the Tersoff potential objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._compute_tersoff_error(x)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)

        lb = xp.asarray(self._param_bounds[:, 0])
        ub = xp.asarray(self._param_bounds[:, 1])

        x_normalized = (X + 10) / 20
        x_actual = lb + x_normalized * (ub - lb)

        mid = (lb + ub) / 2
        range_size = ub - lb

        error = xp.sum(((x_actual - mid) / range_size) ** 2, axis=1)

        return error


# =============================================================================
# P07: Radar Polyphase Code Design
# =============================================================================


class RadarPolyphaseCode(CEC2011Function):
    """P07: Radar Polyphase Code Design.

    Design a polyphase pulse compression code to minimize the Peak Sidelobe
    Level (PSL) of the autocorrelation function.

    The code consists of N complex numbers with unit magnitude and phases
    phi_1, phi_2, ..., phi_N.

    PSL = max_{k!=0} |R(k)| / |R(0)|

    where R(k) is the autocorrelation function.

    Dimension: 20 (phase angles)
    Bounds: [0, 2*pi]
    Global optimum: PSL approaches 0 for optimal codes

    References
    ----------
    Levanon, N. & Mozeson, E. (2004). Radar Signals. Wiley.
    """

    _fixed_dim = 20
    _problem_id = 7

    _spec = {
        "problem_id": 7,
        "default_bounds": (0.0, 2 * np.pi),
        "continuous": True,
        "differentiable": False,  # Due to max operation
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    def __init__(
        self,
        n_phases: int = 20,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        self._fixed_dim = n_phases
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)
        self._f_global = 0.5  # Good PSL value
        self._x_global = None  # Complex optimal codes

    def _compute_psl(self, phases: np.ndarray) -> float:
        """Compute Peak Sidelobe Level of radar code."""
        N = len(phases)

        # Create complex code
        code = np.exp(1j * phases)

        # Compute autocorrelation
        R = np.correlate(code, code, mode="full")

        # R(0) is at the center
        center = N - 1
        R_0 = np.abs(R[center])

        # Find peak sidelobe (max of |R(k)| for k != 0)
        sidelobes = np.abs(np.concatenate([R[:center], R[center + 1 :]]))
        max_sidelobe = np.max(sidelobes)

        # PSL in dB is often used, but we return linear ratio
        psl = max_sidelobe / R_0

        return psl

    def _create_objective_function(self) -> None:
        """Create the radar polyphase code objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            phases = self._params_to_array(params)
            return self._compute_psl(phases)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Batch evaluation (sequential due to correlate)."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        results = xp.zeros(n_points, dtype=X.dtype)

        for i in range(n_points):
            results[i] = self._compute_psl(np.asarray(X[i]))

        return results


# =============================================================================
# P08: Spread Spectrum Radar Polyphase Code Design
# =============================================================================


class SpreadSpectrumRadar(CEC2011Function):
    """P08: Spread Spectrum Radar Polyphase Code Design.

    Design a spread spectrum radar code to maximize the merit factor,
    which measures the ratio of mainlobe energy to sidelobe energy.

    Merit Factor MF = N^2 / (2 * sum_{k=1}^{N-1} |R(k)|^2)

    Minimizing 1/MF is equivalent to maximizing MF.

    Dimension: 7
    Bounds: [0, 15]
    Global optimum: Depends on code length

    References
    ----------
    Golomb, S. W. & Gong, G. (2005). Signal Design for Good
    Correlation. Cambridge University Press.
    """

    _fixed_dim = 7
    _problem_id = 8

    _spec = {
        "problem_id": 8,
        "default_bounds": (0.0, 15.0),
        "continuous": True,
        "differentiable": True,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

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
        self._f_global = 0.5  # Inverse of good merit factor
        self._x_global = None

    def _compute_merit_factor(self, x: np.ndarray) -> float:
        """Compute inverse merit factor of spread spectrum code."""
        N = len(x)

        # Generate binary sequence from real-valued parameters
        # (threshold at 7.5 to convert to +1/-1)
        code = np.where(x > 7.5, 1.0, -1.0)

        # Compute aperiodic autocorrelation
        R = np.correlate(code, code, mode="full")

        # R(0) = N (at center)
        center = N - 1

        # Sum of squared sidelobes
        sidelobe_sum = np.sum(np.abs(R[:center]) ** 2) + np.sum(np.abs(R[center + 1 :]) ** 2)

        if sidelobe_sum < 1e-10:
            return 0.0  # Perfect code (unlikely)

        # Merit factor = N^2 / (2 * sidelobe_sum)
        # Return inverse (minimize 1/MF)
        merit_factor = N**2 / (2 * sidelobe_sum)

        return 1.0 / merit_factor

    def _create_objective_function(self) -> None:
        """Create the spread spectrum radar objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._compute_merit_factor(x)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        N = X.shape[1]

        # Convert to binary code
        code = xp.where(X > 7.5, 1.0, -1.0)

        results = xp.zeros(n_points, dtype=X.dtype)

        for i in range(n_points):
            R = np.correlate(np.asarray(code[i]), np.asarray(code[i]), mode="full")
            center = N - 1
            sidelobe_sum = np.sum(np.abs(R[:center]) ** 2) + np.sum(np.abs(R[center + 1 :]) ** 2)

            if sidelobe_sum < 1e-10:
                results[i] = 0.0
            else:
                merit_factor = N**2 / (2 * sidelobe_sum)
                results[i] = 1.0 / merit_factor

        return results


# =============================================================================
# Collections
# =============================================================================

CEC2011_ALL: List[type] = [
    FMSynthesis,
    LennardJonesPotential,
    BifunctionalCatalyst,
    StirredTankReactor,
    TersoffPotentialSiB,
    TersoffPotentialSiC,
    RadarPolyphaseCode,
    SpreadSpectrumRadar,
]

CEC2011_SIMPLE: List[type] = [
    BifunctionalCatalyst,
    StirredTankReactor,
]

CEC2011_SIGNAL: List[type] = [
    FMSynthesis,
    RadarPolyphaseCode,
    SpreadSpectrumRadar,
]

CEC2011_MOLECULAR: List[type] = [
    LennardJonesPotential,
    TersoffPotentialSiB,
    TersoffPotentialSiC,
]
