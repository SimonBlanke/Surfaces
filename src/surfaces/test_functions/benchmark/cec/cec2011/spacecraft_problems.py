# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2011 spacecraft trajectory optimization problems P12-P13.

These problems are Multiple Gravity Assist with Deep Space Maneuvers (MGA-1DSM)
trajectory optimization problems from the ESA GTOP database.

P12: Cassini 2 - E-V-V-E-J-S trajectory (22 dimensions)
P13: Messenger (full) - E-V-V-M-M-M-M trajectory (26 dimensions)

The implementation uses:
- Universal variable Lambert solver for orbital transfers
- Simplified Keplerian planetary ephemeris
- Delta-V minimization objective

References
----------
Das, S. & Suganthan, P. N. (2010). Problem Definitions and Evaluation
Criteria for CEC 2011 Competition on Testing Evolutionary Algorithms
on Real World Optimization Problems. Technical Report.

Izzo, D. (2015). Revisiting Lambert's problem. Celestial Mechanics and
Dynamical Astronomy, 121(1), 1-15.

ESA Advanced Concepts Team. Global Trajectory Optimization Problems Database.
https://www.esa.int/gsp/ACT/projects/gtop/
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ._base_cec2011 import CEC2011Function

# =============================================================================
# Physical Constants
# =============================================================================

MU_SUN = 1.32712440018e11  # Sun gravitational parameter [km^3/s^2]
AU = 1.49597870691e8  # Astronomical Unit [km]
DAY = 86400.0  # Seconds per day
MJD2000 = 51544.5  # Modified Julian Date of J2000.0


# =============================================================================
# Simplified Planetary Ephemeris (Keplerian Elements)
# =============================================================================


class KeplerianEphemeris:
    """Simplified planetary ephemeris using mean Keplerian elements.

    This provides approximate planet positions suitable for trajectory
    optimization. For production use, JPL DE ephemeris should be used.

    Mean elements at J2000.0 epoch with linear rates.
    Data from NASA JPL Solar System Dynamics.
    """

    # Orbital elements: [a (AU), e, i (deg), Omega (deg), omega (deg), M0 (deg), n (deg/day)]
    # a = semi-major axis, e = eccentricity, i = inclination
    # Omega = longitude of ascending node, omega = argument of perihelion
    # M0 = mean anomaly at J2000, n = mean motion
    PLANETS = {
        "mercury": {
            "a": 0.38709927,
            "e": 0.20563593,
            "i": 7.00497902,
            "Omega": 48.33076593,
            "omega": 77.45779628,
            "L0": 252.25032350,
            "n": 4.09233445,
        },
        "venus": {
            "a": 0.72333566,
            "e": 0.00677672,
            "i": 3.39467605,
            "Omega": 76.67984255,
            "omega": 131.60246718,
            "L0": 181.97909950,
            "n": 1.60213034,
        },
        "earth": {
            "a": 1.00000261,
            "e": 0.01671123,
            "i": -0.00001531,
            "Omega": 0.0,
            "omega": 102.93768193,
            "L0": 100.46457166,
            "n": 0.98560028,
        },
        "mars": {
            "a": 1.52371034,
            "e": 0.09339410,
            "i": 1.84969142,
            "Omega": 49.55953891,
            "omega": 336.05637041,
            "L0": -4.55343205,
            "n": 0.52402068,
        },
        "jupiter": {
            "a": 5.20288700,
            "e": 0.04838624,
            "i": 1.30439695,
            "Omega": 100.47390909,
            "omega": 14.72847983,
            "L0": 34.39644051,
            "n": 0.08308529,
        },
        "saturn": {
            "a": 9.53667594,
            "e": 0.05386179,
            "i": 2.48599187,
            "Omega": 113.66242448,
            "omega": 92.59887831,
            "L0": 49.95424423,
            "n": 0.03344414,
        },
    }

    # Planetary gravitational parameters [km^3/s^2]
    MU = {
        "mercury": 2.2032e4,
        "venus": 3.24859e5,
        "earth": 3.986004418e5,
        "mars": 4.282837e4,
        "jupiter": 1.26686534e8,
        "saturn": 3.7931187e7,
    }

    # Planetary radii [km] (for flyby calculations)
    RADIUS = {
        "mercury": 2439.7,
        "venus": 6051.8,
        "earth": 6371.0,
        "mars": 3389.5,
        "jupiter": 69911.0,
        "saturn": 58232.0,
    }

    @classmethod
    def position_velocity(cls, planet: str, mjd2000: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get heliocentric position and velocity of a planet.

        Parameters
        ----------
        planet : str
            Planet name (lowercase)
        mjd2000 : float
            Modified Julian Date relative to J2000 (days since 2000-01-01)

        Returns
        -------
        r : np.ndarray
            Heliocentric position [km], shape (3,)
        v : np.ndarray
            Heliocentric velocity [km/s], shape (3,)
        """
        elem = cls.PLANETS[planet.lower()]

        # Mean longitude at time t
        L = elem["L0"] + elem["n"] * mjd2000
        L = np.deg2rad(L % 360.0)

        # Mean anomaly
        omega = np.deg2rad(elem["omega"])
        M = L - omega

        # Solve Kepler's equation for eccentric anomaly
        e = elem["e"]
        E = cls._solve_kepler(M, e)

        # True anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

        # Distance
        a = elem["a"] * AU  # Convert to km
        r_mag = a * (1 - e * np.cos(E))

        # Orbital elements in radians
        i = np.deg2rad(elem["i"])
        Omega = np.deg2rad(elem["Omega"])
        omega_arg = omega - Omega  # Argument of perihelion

        # Position in orbital plane
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)

        # Rotation matrices
        cos_O, sin_O = np.cos(Omega), np.sin(Omega)
        cos_i, sin_i = np.cos(i), np.sin(i)
        cos_w, sin_w = np.cos(omega_arg), np.sin(omega_arg)

        # Position in heliocentric ecliptic frame
        x_orb = r_mag * cos_nu
        y_orb = r_mag * sin_nu

        r = np.array(
            [
                (cos_O * cos_w - sin_O * sin_w * cos_i) * x_orb + (-cos_O * sin_w - sin_O * cos_w * cos_i) * y_orb,
                (sin_O * cos_w + cos_O * sin_w * cos_i) * x_orb + (-sin_O * sin_w + cos_O * cos_w * cos_i) * y_orb,
                (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb,
            ]
        )

        # Velocity
        n = np.sqrt(MU_SUN / a**3)  # Mean motion [rad/s]
        vx_orb = -n * a * np.sin(E) / (1 - e * np.cos(E)) * AU / DAY
        vy_orb = n * a * np.sqrt(1 - e**2) * np.cos(E) / (1 - e * np.cos(E)) * AU / DAY

        # Actually compute velocity properly
        p = a * (1 - e**2)
        h = np.sqrt(MU_SUN * p)
        vx_orb = -MU_SUN / h * np.sin(nu)
        vy_orb = MU_SUN / h * (e + np.cos(nu))

        v = np.array(
            [
                (cos_O * cos_w - sin_O * sin_w * cos_i) * vx_orb + (-cos_O * sin_w - sin_O * cos_w * cos_i) * vy_orb,
                (sin_O * cos_w + cos_O * sin_w * cos_i) * vx_orb + (-sin_O * sin_w + cos_O * cos_w * cos_i) * vy_orb,
                (sin_w * sin_i) * vx_orb + (cos_w * sin_i) * vy_orb,
            ]
        )

        return r, v

    @staticmethod
    def _solve_kepler(M: float, e: float, tol: float = 1e-10) -> float:
        """Solve Kepler's equation M = E - e*sin(E) for E.

        Uses Newton-Raphson iteration.
        """
        # Normalize M to [-pi, pi]
        M = np.mod(M + np.pi, 2 * np.pi) - np.pi

        # Initial guess
        if e < 0.8:
            E = M
        else:
            E = np.pi * np.sign(M)

        # Newton-Raphson
        for _ in range(50):
            f = E - e * np.sin(E) - M
            fp = 1 - e * np.cos(E)
            dE = -f / fp
            E = E + dE
            if abs(dE) < tol:
                break

        return E


# =============================================================================
# Lambert Problem Solver
# =============================================================================


class LambertSolver:
    """Universal variable Lambert problem solver.

    Solves the two-point boundary value problem to find initial and final
    velocities given two position vectors and transfer time.

    Uses the universal variable formulation which handles all orbit types
    (elliptic, parabolic, hyperbolic).

    References
    ----------
    Battin, R.H. (1999). An Introduction to the Mathematics and Methods
    of Astrodynamics. AIAA.

    Izzo, D. (2015). Revisiting Lambert's problem.
    """

    @staticmethod
    def solve(
        r1: np.ndarray,
        r2: np.ndarray,
        tof: float,
        mu: float = MU_SUN,
        prograde: bool = True,
        max_iter: int = 50,
        tol: float = 1e-10,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Solve Lambert's problem.

        Parameters
        ----------
        r1 : np.ndarray
            Initial position vector [km], shape (3,)
        r2 : np.ndarray
            Final position vector [km], shape (3,)
        tof : float
            Time of flight [s]
        mu : float
            Gravitational parameter [km^3/s^2]
        prograde : bool
            True for prograde orbit (short way), False for retrograde
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance

        Returns
        -------
        v1 : np.ndarray
            Initial velocity [km/s]
        v2 : np.ndarray
            Final velocity [km/s]
        converged : bool
            True if solution converged
        """
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)

        # Cross product to determine orbit plane
        h = np.cross(r1, r2)

        # True anomaly change
        cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
        cos_dnu = np.clip(cos_dnu, -1.0, 1.0)

        # Determine direction
        if prograde:
            if h[2] >= 0:
                sin_dnu = np.sqrt(1 - cos_dnu**2)
            else:
                sin_dnu = -np.sqrt(1 - cos_dnu**2)
        else:
            if h[2] < 0:
                sin_dnu = np.sqrt(1 - cos_dnu**2)
            else:
                sin_dnu = -np.sqrt(1 - cos_dnu**2)

        dnu = np.arctan2(sin_dnu, cos_dnu)

        # Chord
        c = np.sqrt(r1_mag**2 + r2_mag**2 - 2 * r1_mag * r2_mag * cos_dnu)

        # Semi-perimeter
        s = (r1_mag + r2_mag + c) / 2

        # Minimum energy semi-major axis
        a_m = s / 2

        # Minimum energy time of flight
        alpha_m = 2 * np.arcsin(np.sqrt(s / (2 * a_m)))
        beta_m = 2 * np.arcsin(np.sqrt((s - c) / (2 * a_m)))

        if dnu > np.pi:
            beta_m = -beta_m

        # Use universal variable formulation
        # Initial guess based on geometry
        if tof < 1e-6:
            return np.zeros(3), np.zeros(3), False

        # Parabolic TOF
        parab_tof = (1.0 / 3.0) * np.sqrt(2 / mu) * (s**1.5 - np.sign(sin_dnu) * (s - c) ** 1.5)

        # Initial guess for universal variable z
        if tof < parab_tof:
            # Hyperbolic
            z = -1.0
        else:
            # Elliptic
            z = (np.pi / 2) ** 2

        # Stumpff functions
        def stumpff_c(z):
            if z > 1e-6:
                sz = np.sqrt(z)
                return (1 - np.cos(sz)) / z
            elif z < -1e-6:
                sz = np.sqrt(-z)
                return (1 - np.cosh(sz)) / z
            else:
                return 1 / 2 - z / 24 + z**2 / 720

        def stumpff_s(z):
            if z > 1e-6:
                sz = np.sqrt(z)
                return (sz - np.sin(sz)) / (sz**3)
            elif z < -1e-6:
                sz = np.sqrt(-z)
                return (np.sinh(sz) - sz) / (sz**3)
            else:
                return 1 / 6 - z / 120 + z**2 / 5040

        # A parameter (constant for given geometry)
        A = sin_dnu * np.sqrt(r1_mag * r2_mag / (1 - cos_dnu))

        # Newton-Raphson iteration
        converged = False
        for _ in range(max_iter):
            C = stumpff_c(z)
            S = stumpff_s(z)

            # y function
            y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)

            if y < 0:
                # Invalid (need to adjust z)
                z = z * 0.5
                continue

            # x (universal anomaly)
            x = np.sqrt(y / C)

            # Time of flight for this z
            t = (x**3 * S + A * np.sqrt(y)) / np.sqrt(mu)

            # Derivative dt/dz
            if abs(z) > 1e-6:
                dt_dz = (x**3 * (stumpff_s(z) - 3 * S / (2 * z)) / C + A / 8 * (3 * S * np.sqrt(y) / C + A / x)) / np.sqrt(mu)
            else:
                dt_dz = (
                    np.sqrt(2) / 40 * y**1.5 + A / 8 * (np.sqrt(y) + A * np.sqrt(1 / (2 * y)))
                ) / np.sqrt(mu)

            # Update z
            dz = (tof - t) / dt_dz
            z = z + dz

            if abs(dz) < tol:
                converged = True
                break

        if not converged:
            return np.zeros(3), np.zeros(3), False

        # Compute final values
        C = stumpff_c(z)
        S = stumpff_s(z)
        y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)

        # Lagrange coefficients
        f = 1 - y / r1_mag
        g = A * np.sqrt(y / mu)
        g_dot = 1 - y / r2_mag

        # Velocities
        v1 = (r2 - f * r1) / g
        v2 = (g_dot * r2 - r1) / g

        return v1, v2, True


# =============================================================================
# MGA-1DSM Trajectory Model
# =============================================================================


def compute_mga1dsm_deltav(
    decision_vars: np.ndarray,
    sequence: List[str],
    bounds: np.ndarray,
) -> float:
    """Compute total delta-V for MGA-1DSM trajectory.

    The decision variables encode:
    - t0: departure epoch (MJD2000)
    - vinf: departure hyperbolic excess velocity magnitude
    - u, v: direction of departure vinf (spherical coordinates)
    - For each leg (i=1 to n_legs):
        - TOF_i: time of flight
        - eta_i: fraction of TOF at which DSM occurs (0 to 1)
        - Rp_i: periapsis radius of flyby (as ratio to planet radius)
        - beta_i: B-plane angle for flyby

    Parameters
    ----------
    decision_vars : np.ndarray
        Decision variables (varies by sequence)
    sequence : List[str]
        Sequence of planets (e.g., ["earth", "venus", "venus", "earth", "jupiter", "saturn"])
    bounds : np.ndarray
        Variable bounds, shape (n_vars, 2)

    Returns
    -------
    float
        Total delta-V [km/s] (lower is better)
    """
    n_legs = len(sequence) - 1

    # Parse decision variables
    t0 = decision_vars[0]  # Departure epoch
    vinf_mag = decision_vars[1]  # Departure Vinf magnitude
    u = decision_vars[2]  # Vinf direction parameter 1
    v = decision_vars[3]  # Vinf direction parameter 2

    # Compute departure Vinf direction (unit sphere parametrization)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    vinf_dir = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])

    # Get departure planet state
    r_dep, v_dep = KeplerianEphemeris.position_velocity(sequence[0], t0)

    # Departure velocity (planet velocity + hyperbolic excess)
    v_sc = v_dep + vinf_mag * vinf_dir

    total_deltav = 0.0
    current_time = t0

    # Process each leg
    var_idx = 4  # Current index in decision variables
    r_current = r_dep
    v_current = v_sc

    for leg in range(n_legs):
        # Leg parameters
        tof = decision_vars[var_idx]  # Time of flight [days]
        var_idx += 1

        # Get target planet state
        arrival_time = current_time + tof
        r_target, v_target = KeplerianEphemeris.position_velocity(sequence[leg + 1], arrival_time)

        # Solve Lambert problem
        v1, v2, converged = LambertSolver.solve(r_current, r_target, tof * DAY, MU_SUN, prograde=True)

        if not converged:
            return 1e10  # Penalty for non-convergence

        # Deep space maneuver
        # For simplicity, assume DSM at departure to match Lambert solution
        dsm_deltav = np.linalg.norm(v1 - v_current)
        total_deltav += dsm_deltav

        # Relative velocity at arrival
        v_inf_arr = v2 - v_target

        # If not final leg, perform flyby
        if leg < n_legs - 1:
            # Flyby parameters
            rp_ratio = decision_vars[var_idx]  # Periapsis ratio
            beta = decision_vars[var_idx + 1]  # B-plane angle
            var_idx += 2

            target_planet = sequence[leg + 1]
            mu_planet = KeplerianEphemeris.MU.get(target_planet, 1e5)
            r_planet = KeplerianEphemeris.RADIUS.get(target_planet, 6000)

            # Periapsis radius
            rp = rp_ratio * r_planet

            # Flyby hyperbola
            v_inf_mag = np.linalg.norm(v_inf_arr)

            # Turn angle from flyby
            # delta = 2 * arcsin(1 / e), where e = 1 + rp * v_inf^2 / mu
            if v_inf_mag > 1e-6:
                e = 1 + rp * v_inf_mag**2 / mu_planet
                if e > 1:
                    delta = 2 * np.arcsin(1 / e)
                else:
                    delta = np.pi  # Maximum turn

                # Rotate v_inf by turn angle around B-plane
                # Simplified: rotate in the plane perpendicular to planet velocity
                v_hat = v_inf_arr / v_inf_mag

                # Create rotation axis (simplified - perpendicular to velocity)
                h = np.cross(r_target, v_target)
                h = h / np.linalg.norm(h)

                # Rodrigues rotation
                cos_d = np.cos(delta)
                sin_d = np.sin(delta)

                v_inf_dep = v_inf_arr * cos_d + np.cross(h, v_inf_arr) * sin_d + h * np.dot(h, v_inf_arr) * (1 - cos_d)

                # Apply B-plane rotation
                cos_b = np.cos(beta)
                sin_b = np.sin(beta)
                v_inf_dep = v_inf_dep * cos_b + np.cross(v_hat, v_inf_dep) * sin_b

                # New spacecraft velocity
                v_current = v_target + v_inf_dep
            else:
                v_current = v_target

            r_current = r_target
        else:
            # Final leg - arrival delta-V
            # For Cassini/Messenger, we capture into orbit
            arrival_deltav = np.linalg.norm(v_inf_arr)
            total_deltav += arrival_deltav

        current_time = arrival_time

    return total_deltav


# =============================================================================
# P12: Cassini 2 Trajectory
# =============================================================================


class Cassini2(CEC2011Function):
    """P12: Cassini 2 Spacecraft Trajectory Optimization.

    Optimize the trajectory of the Cassini spacecraft from Earth to Saturn
    using the sequence: Earth - Venus - Venus - Earth - Jupiter - Saturn
    (E-V-V-E-J-S).

    The problem uses the MGA-1DSM (Multiple Gravity Assist with one Deep
    Space Maneuver per leg) trajectory model.

    Decision variables (22 total):
    - t0: departure date [MJD2000]
    - Vinf: departure excess velocity [km/s]
    - u, v: Vinf direction parameters
    - For each of 5 legs: TOF, and for non-final legs: Rp_ratio, beta

    Dimension: 22
    Bounds: Problem-specific
    Global optimum: Approximately 8.4 km/s total delta-V

    References
    ----------
    ESA GTOP Database - Cassini 2 trajectory.
    """

    _fixed_dim = 22
    _problem_id = 12

    _spec = {
        "problem_id": 12,
        "default_bounds": (0.0, 1.0),  # Normalized, actual bounds applied internally
        "continuous": True,
        "differentiable": False,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    # Planet sequence for Cassini 2
    _sequence = ["earth", "venus", "venus", "earth", "jupiter", "saturn"]

    # Variable bounds from ESA GTOP
    # [t0, Vinf, u, v, TOF1, Rp1, beta1, TOF2, Rp2, beta2, TOF3, Rp3, beta3, TOF4, Rp4, beta4, TOF5]
    _bounds = np.array(
        [
            [-1000.0, 0.0],  # t0: MJD2000 (before 2000)
            [3.0, 5.0],  # Vinf: km/s
            [0.0, 1.0],  # u
            [0.0, 1.0],  # v
            [100.0, 400.0],  # TOF1: E-V
            [1.05, 10.0],  # Rp1: Venus flyby
            [-np.pi, np.pi],  # beta1
            [100.0, 500.0],  # TOF2: V-V
            [1.05, 10.0],  # Rp2: Venus flyby
            [-np.pi, np.pi],  # beta2
            [30.0, 300.0],  # TOF3: V-E
            [1.05, 10.0],  # Rp3: Earth flyby
            [-np.pi, np.pi],  # beta3
            [400.0, 1600.0],  # TOF4: E-J
            [1.05, 100.0],  # Rp4: Jupiter flyby
            [-np.pi, np.pi],  # beta4
            [800.0, 2200.0],  # TOF5: J-S (final, no flyby params)
            # Additional vars for completeness (22 total)
            [0.0, 1.0],  # eta1
            [0.0, 1.0],  # eta2
            [0.0, 1.0],  # eta3
            [0.0, 1.0],  # eta4
            [0.0, 1.0],  # eta5
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
        self._f_global = 8.4  # Approximate best known delta-V [km/s]
        self._x_global = None

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Convert normalized [0,1] variables to actual bounds."""
        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]
        return lb + x * (ub - lb)

    def _compute_trajectory(self, x: np.ndarray) -> float:
        """Compute total delta-V for trajectory.

        Parameters
        ----------
        x : np.ndarray
            Normalized decision variables [0, 1]

        Returns
        -------
        float
            Total delta-V [km/s]
        """
        # Denormalize
        x_actual = self._denormalize(x)

        # Extract variables
        t0 = x_actual[0]
        vinf = x_actual[1]
        u = x_actual[2]
        v = x_actual[3]

        # Build simplified decision vector for trajectory computation
        # [t0, vinf, u, v, TOF1, TOF2, ..., TOF5, rp1, beta1, rp2, beta2, ...]
        decision = np.zeros(22)
        decision[0] = t0
        decision[1] = vinf
        decision[2] = u
        decision[3] = v

        # TOFs
        decision[4] = x_actual[4]  # TOF1
        decision[5] = x_actual[7]  # TOF2
        decision[6] = x_actual[10]  # TOF3
        decision[7] = x_actual[13]  # TOF4
        decision[8] = x_actual[16]  # TOF5

        # Flyby params
        decision[9] = x_actual[5]  # Rp1
        decision[10] = x_actual[6]  # beta1
        decision[11] = x_actual[8]  # Rp2
        decision[12] = x_actual[9]  # beta2
        decision[13] = x_actual[11]  # Rp3
        decision[14] = x_actual[12]  # beta3
        decision[15] = x_actual[14]  # Rp4
        decision[16] = x_actual[15]  # beta4

        # Simplified computation
        try:
            deltav = self._compute_deltav_simple(x_actual)
        except Exception:
            deltav = 1e10

        return deltav

    def _compute_deltav_simple(self, x: np.ndarray) -> float:
        """Simplified delta-V computation."""
        t0 = x[0]
        vinf = x[1]
        u, v = x[2], x[3]

        # Departure
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        vinf_dir = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])

        total_dv = 0.0
        current_time = t0

        r_dep, v_dep = KeplerianEphemeris.position_velocity(self._sequence[0], t0)
        v_sc = v_dep + vinf * vinf_dir

        # Index for TOF and flyby params
        idx_tof = [4, 7, 10, 13, 16]
        idx_rp = [5, 8, 11, 14]
        idx_beta = [6, 9, 12, 15]

        r_current = r_dep
        v_current = v_sc

        for leg in range(5):
            tof = x[idx_tof[leg]]
            arrival_time = current_time + tof

            # Target planet
            target = self._sequence[leg + 1]
            r_target, v_target = KeplerianEphemeris.position_velocity(target, arrival_time)

            # Lambert
            v1, v2, conv = LambertSolver.solve(r_current, r_target, tof * DAY, MU_SUN)

            if not conv:
                return 1e10

            # DSM
            dsm = np.linalg.norm(v1 - v_current)
            total_dv += dsm

            # Arrival Vinf
            v_inf_arr = v2 - v_target
            v_inf_mag = np.linalg.norm(v_inf_arr)

            if leg < 4:
                # Flyby
                rp_ratio = x[idx_rp[leg]]
                beta = x[idx_beta[leg]]

                mu_p = KeplerianEphemeris.MU.get(target, 1e5)
                r_p = KeplerianEphemeris.RADIUS.get(target, 6000)
                rp = rp_ratio * r_p

                if v_inf_mag > 1e-6:
                    e = 1 + rp * v_inf_mag**2 / mu_p
                    if e > 1:
                        delta = 2 * np.arcsin(1 / e)
                    else:
                        delta = np.pi

                    # Rotate
                    h = np.cross(r_target, v_target)
                    h_norm = np.linalg.norm(h)
                    if h_norm > 1e-10:
                        h = h / h_norm
                    else:
                        h = np.array([0, 0, 1])

                    cd, sd = np.cos(delta), np.sin(delta)
                    v_out = v_inf_arr * cd + np.cross(h, v_inf_arr) * sd + h * np.dot(h, v_inf_arr) * (1 - cd)

                    # B-plane rotation
                    v_hat = v_inf_arr / v_inf_mag
                    cb, sb = np.cos(beta), np.sin(beta)
                    v_out = v_out * cb + np.cross(v_hat, v_out) * sb

                    v_current = v_target + v_out
                else:
                    v_current = v_target
            else:
                # Final arrival
                total_dv += v_inf_mag

            r_current = r_target
            current_time = arrival_time

        return total_dv

    def _create_objective_function(self) -> None:
        """Create the Cassini 2 objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._compute_trajectory(x)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Batch evaluation (sequential due to orbital mechanics)."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        results = xp.zeros(n_points, dtype=X.dtype)

        for i in range(n_points):
            results[i] = self._compute_trajectory(np.asarray(X[i]))

        return results


# =============================================================================
# P13: Messenger (Full) Trajectory
# =============================================================================


class Messenger(CEC2011Function):
    """P13: Messenger Spacecraft Trajectory Optimization (Full).

    Optimize the trajectory of the Messenger spacecraft from Earth to Mercury
    using the sequence: Earth - Venus - Venus - Mercury - Mercury - Mercury - Mercury
    (E-V-V-M-M-M-M with deep space maneuvers).

    The problem uses the MGA-1DSM trajectory model with multiple Mercury flybys
    to reduce arrival velocity at Mercury.

    Decision variables (26 total):
    - t0: departure date [MJD2000]
    - Vinf: departure excess velocity [km/s]
    - u, v: Vinf direction parameters
    - For each of 6 legs: TOF, and for non-final legs: Rp_ratio, beta
    - eta values for DSM timing

    Dimension: 26
    Bounds: Problem-specific
    Global optimum: Approximately 8.6 km/s total delta-V

    References
    ----------
    ESA GTOP Database - Messenger (full) trajectory.
    """

    _fixed_dim = 26
    _problem_id = 13

    _spec = {
        "problem_id": 13,
        "default_bounds": (0.0, 1.0),
        "continuous": True,
        "differentiable": False,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    # Planet sequence for Messenger
    _sequence = ["earth", "venus", "venus", "mercury", "mercury", "mercury", "mercury"]

    # Variable bounds from ESA GTOP
    _bounds = np.array(
        [
            [1900.0, 2300.0],  # t0: MJD2000
            [2.5, 4.05],  # Vinf: km/s
            [0.0, 1.0],  # u
            [0.0, 1.0],  # v
            [100.0, 500.0],  # TOF1: E-V
            [1.05, 10.0],  # Rp1: Venus flyby
            [-np.pi, np.pi],  # beta1
            [50.0, 400.0],  # TOF2: V-V
            [1.05, 10.0],  # Rp2: Venus flyby
            [-np.pi, np.pi],  # beta2
            [50.0, 300.0],  # TOF3: V-M
            [1.05, 10.0],  # Rp3: Mercury flyby 1
            [-np.pi, np.pi],  # beta3
            [50.0, 300.0],  # TOF4: M-M
            [1.05, 10.0],  # Rp4: Mercury flyby 2
            [-np.pi, np.pi],  # beta4
            [50.0, 300.0],  # TOF5: M-M
            [1.05, 10.0],  # Rp5: Mercury flyby 3
            [-np.pi, np.pi],  # beta5
            [50.0, 300.0],  # TOF6: M-M (final)
            [0.0, 1.0],  # eta1
            [0.0, 1.0],  # eta2
            [0.0, 1.0],  # eta3
            [0.0, 1.0],  # eta4
            [0.0, 1.0],  # eta5
            [0.0, 1.0],  # eta6
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
        self._f_global = 8.6  # Approximate best known delta-V [km/s]
        self._x_global = None

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Convert normalized [0,1] variables to actual bounds."""
        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]
        return lb + x * (ub - lb)

    def _compute_trajectory(self, x: np.ndarray) -> float:
        """Compute total delta-V for trajectory."""
        x_actual = self._denormalize(x)

        t0 = x_actual[0]
        vinf = x_actual[1]
        u, v = x_actual[2], x_actual[3]

        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        vinf_dir = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])

        total_dv = 0.0
        current_time = t0

        r_dep, v_dep = KeplerianEphemeris.position_velocity(self._sequence[0], t0)
        v_sc = v_dep + vinf * vinf_dir

        # Indices for parameters
        # TOF: 4, 7, 10, 13, 16, 19
        # Rp: 5, 8, 11, 14, 17
        # beta: 6, 9, 12, 15, 18
        idx_tof = [4, 7, 10, 13, 16, 19]
        idx_rp = [5, 8, 11, 14, 17]
        idx_beta = [6, 9, 12, 15, 18]

        r_current = r_dep
        v_current = v_sc

        n_legs = 6

        for leg in range(n_legs):
            tof = x_actual[idx_tof[leg]]
            arrival_time = current_time + tof

            target = self._sequence[leg + 1]
            r_target, v_target = KeplerianEphemeris.position_velocity(target, arrival_time)

            v1, v2, conv = LambertSolver.solve(r_current, r_target, tof * DAY, MU_SUN)

            if not conv:
                return 1e10

            dsm = np.linalg.norm(v1 - v_current)
            total_dv += dsm

            v_inf_arr = v2 - v_target
            v_inf_mag = np.linalg.norm(v_inf_arr)

            if leg < n_legs - 1:
                rp_ratio = x_actual[idx_rp[leg]]
                beta = x_actual[idx_beta[leg]]

                mu_p = KeplerianEphemeris.MU.get(target, 1e5)
                r_p = KeplerianEphemeris.RADIUS.get(target, 6000)
                rp = rp_ratio * r_p

                if v_inf_mag > 1e-6:
                    e = 1 + rp * v_inf_mag**2 / mu_p
                    if e > 1:
                        delta = 2 * np.arcsin(1 / e)
                    else:
                        delta = np.pi

                    h = np.cross(r_target, v_target)
                    h_norm = np.linalg.norm(h)
                    if h_norm > 1e-10:
                        h = h / h_norm
                    else:
                        h = np.array([0, 0, 1])

                    cd, sd = np.cos(delta), np.sin(delta)
                    v_out = v_inf_arr * cd + np.cross(h, v_inf_arr) * sd + h * np.dot(h, v_inf_arr) * (1 - cd)

                    v_hat = v_inf_arr / v_inf_mag
                    cb, sb = np.cos(beta), np.sin(beta)
                    v_out = v_out * cb + np.cross(v_hat, v_out) * sb

                    v_current = v_target + v_out
                else:
                    v_current = v_target
            else:
                total_dv += v_inf_mag

            r_current = r_target
            current_time = arrival_time

        return total_dv

    def _create_objective_function(self) -> None:
        """Create the Messenger objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._compute_trajectory(x)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        results = xp.zeros(n_points, dtype=X.dtype)

        for i in range(n_points):
            results[i] = self._compute_trajectory(np.asarray(X[i]))

        return results


# =============================================================================
# P14: Cassini 1 (MGA without DSM)
# =============================================================================


class Cassini1(CEC2011Function):
    """P14: Cassini 1 Spacecraft Trajectory (MGA).

    Optimize the Cassini trajectory using pure Multiple Gravity Assist (MGA)
    without deep space maneuvers. This is a simpler model than Cassini 2.

    Sequence: Earth - Venus - Venus - Earth - Jupiter - Saturn (E-V-V-E-J-S)

    Decision variables (6 total):
    - t0: departure date [MJD2000]
    - TOF1-TOF5: time of flight for each leg [days]

    The spacecraft uses only planetary flybys for trajectory shaping,
    with no intermediate propulsive maneuvers.

    Dimension: 6
    Bounds: Problem-specific
    Global optimum: Approximately 4.93 km/s total delta-V

    References
    ----------
    ESA GTOP Database - Cassini 1 trajectory.
    """

    _fixed_dim = 6
    _problem_id = 14

    _spec = {
        "problem_id": 14,
        "default_bounds": (0.0, 1.0),
        "continuous": True,
        "differentiable": False,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    _sequence = ["earth", "venus", "venus", "earth", "jupiter", "saturn"]

    # Variable bounds from ESA GTOP
    _bounds = np.array(
        [
            [-1000.0, 0.0],  # t0: MJD2000
            [30.0, 400.0],  # TOF1: E-V
            [100.0, 470.0],  # TOF2: V-V
            [30.0, 400.0],  # TOF3: V-E
            [400.0, 2000.0],  # TOF4: E-J
            [1000.0, 6000.0],  # TOF5: J-S
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
        self._f_global = 4.93  # Best known delta-V [km/s]
        self._x_global = None

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Convert normalized [0,1] variables to actual bounds."""
        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]
        return lb + x * (ub - lb)

    def _compute_trajectory(self, x: np.ndarray) -> float:
        """Compute total delta-V for MGA trajectory."""
        x_actual = self._denormalize(x)

        t0 = x_actual[0]
        tofs = x_actual[1:6]

        total_dv = 0.0
        current_time = t0

        # Get departure state
        r_dep, v_dep = KeplerianEphemeris.position_velocity(self._sequence[0], t0)

        # First leg: Earth to Venus
        arrival_time = current_time + tofs[0]
        r_target, v_target = KeplerianEphemeris.position_velocity(self._sequence[1], arrival_time)

        v1, v2, conv = LambertSolver.solve(r_dep, r_target, tofs[0] * DAY, MU_SUN)
        if not conv:
            return 1e10

        # Departure delta-V (escape from Earth)
        v_inf_dep = np.linalg.norm(v1 - v_dep)
        total_dv += v_inf_dep

        v_inf_arr = v2 - v_target
        r_current = r_target
        current_time = arrival_time

        # Intermediate legs with flybys
        for leg in range(1, 5):
            # Get next target
            arrival_time = current_time + tofs[leg]
            r_next, v_next = KeplerianEphemeris.position_velocity(self._sequence[leg + 1], arrival_time)

            # Solve Lambert for this leg
            v1_new, v2_new, conv = LambertSolver.solve(r_current, r_next, tofs[leg] * DAY, MU_SUN)
            if not conv:
                return 1e10

            # Check flyby feasibility: v_inf in must match v_inf out (approximately)
            v_inf_out = v1_new - v_target
            v_inf_in_mag = np.linalg.norm(v_inf_arr)
            v_inf_out_mag = np.linalg.norm(v_inf_out)

            # Flyby constraint: |v_inf_in| should equal |v_inf_out|
            # Add penalty for mismatch (this is a simplified constraint)
            if abs(v_inf_in_mag - v_inf_out_mag) > 0.5:
                # Flyby not feasible, add powered flyby delta-V
                total_dv += abs(v_inf_in_mag - v_inf_out_mag)

            # Update for next leg
            v_inf_arr = v2_new - v_next
            v_target = v_next
            r_current = r_next
            current_time = arrival_time

        # Final arrival delta-V (capture at Saturn)
        total_dv += np.linalg.norm(v_inf_arr)

        return total_dv

    def _create_objective_function(self) -> None:
        """Create the Cassini 1 objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._compute_trajectory(x)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        results = xp.zeros(n_points, dtype=X.dtype)

        for i in range(n_points):
            results[i] = self._compute_trajectory(np.asarray(X[i]))

        return results


# =============================================================================
# P15: GTOC1 (Global Trajectory Optimization Competition 1)
# =============================================================================


class GTOC1(CEC2011Function):
    """P15: GTOC1 Asteroid Tour Trajectory.

    Global Trajectory Optimization Competition 1: maximize mass delivered
    to asteroid TW229 using Earth gravity assists.

    Sequence: Earth - Earth - Asteroid TW229 (E-E-A)

    Decision variables (8 total):
    - t0: departure date [MJD2000]
    - TOF1: E-E transfer time
    - TOF2: E-A transfer time
    - Vinf, u, v: departure velocity
    - Rp: Earth flyby periapsis
    - beta: B-plane angle

    Dimension: 8
    Bounds: Problem-specific
    Global optimum: Minimize fuel (maximize final mass)

    References
    ----------
    ESA GTOP Database - GTOC1.
    """

    _fixed_dim = 8
    _problem_id = 15

    _spec = {
        "problem_id": 15,
        "default_bounds": (0.0, 1.0),
        "continuous": True,
        "differentiable": False,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    # Asteroid TW229 orbital elements (simplified)
    _asteroid_elements = {
        "a": 2.5897,  # AU
        "e": 0.5258,
        "i": 7.5046,  # deg
        "Omega": 229.3245,  # deg
        "omega": 264.6829,  # deg
        "L0": 318.5149,  # deg
        "n": 0.24598,  # deg/day
    }

    _bounds = np.array(
        [
            [3000.0, 10000.0],  # t0: MJD2000
            [14.0, 2000.0],  # TOF1: E-E
            [14.0, 2000.0],  # TOF2: E-A
            [0.0, 4.0],  # Vinf: km/s
            [0.0, 1.0],  # u
            [0.0, 1.0],  # v
            [1.05, 10.0],  # Rp: Earth flyby periapsis ratio
            [-np.pi, np.pi],  # beta: B-plane angle
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
        self._f_global = 5.0  # Approximate best delta-V
        self._x_global = None

    def _get_asteroid_position(self, mjd2000: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get asteroid position and velocity."""
        elem = self._asteroid_elements

        L = elem["L0"] + elem["n"] * mjd2000
        L = np.deg2rad(L % 360.0)

        omega = np.deg2rad(elem["omega"])
        M = L - omega

        e = elem["e"]
        E = KeplerianEphemeris._solve_kepler(M, e)

        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

        a = elem["a"] * AU
        r_mag = a * (1 - e * np.cos(E))

        i = np.deg2rad(elem["i"])
        Omega = np.deg2rad(elem["Omega"])
        omega_arg = omega - Omega

        cos_nu, sin_nu = np.cos(nu), np.sin(nu)
        cos_O, sin_O = np.cos(Omega), np.sin(Omega)
        cos_i, sin_i = np.cos(i), np.sin(i)
        cos_w, sin_w = np.cos(omega_arg), np.sin(omega_arg)

        x_orb = r_mag * cos_nu
        y_orb = r_mag * sin_nu

        r = np.array(
            [
                (cos_O * cos_w - sin_O * sin_w * cos_i) * x_orb + (-cos_O * sin_w - sin_O * cos_w * cos_i) * y_orb,
                (sin_O * cos_w + cos_O * sin_w * cos_i) * x_orb + (-sin_O * sin_w + cos_O * cos_w * cos_i) * y_orb,
                (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb,
            ]
        )

        p = a * (1 - e**2)
        h = np.sqrt(MU_SUN * p)
        vx_orb = -MU_SUN / h * np.sin(nu)
        vy_orb = MU_SUN / h * (e + np.cos(nu))

        v = np.array(
            [
                (cos_O * cos_w - sin_O * sin_w * cos_i) * vx_orb + (-cos_O * sin_w - sin_O * cos_w * cos_i) * vy_orb,
                (sin_O * cos_w + cos_O * sin_w * cos_i) * vx_orb + (-sin_O * sin_w + cos_O * cos_w * cos_i) * vy_orb,
                (sin_w * sin_i) * vx_orb + (cos_w * sin_i) * vy_orb,
            ]
        )

        return r, v

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Convert normalized [0,1] variables to actual bounds."""
        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]
        return lb + x * (ub - lb)

    def _compute_trajectory(self, x: np.ndarray) -> float:
        """Compute delta-V for GTOC1 trajectory."""
        x_actual = self._denormalize(x)

        t0 = x_actual[0]
        tof1 = x_actual[1]
        tof2 = x_actual[2]
        vinf = x_actual[3]
        u, v = x_actual[4], x_actual[5]
        rp_ratio = x_actual[6]
        beta = x_actual[7]

        # Departure direction
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        vinf_dir = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])

        total_dv = vinf  # Departure delta-V

        # Earth departure
        r_earth, v_earth = KeplerianEphemeris.position_velocity("earth", t0)
        v_sc = v_earth + vinf * vinf_dir

        # Earth-Earth leg
        t1 = t0 + tof1
        r_earth2, v_earth2 = KeplerianEphemeris.position_velocity("earth", t1)

        v1, v2, conv = LambertSolver.solve(r_earth, r_earth2, tof1 * DAY, MU_SUN)
        if not conv:
            return 1e10

        # DSM for first leg
        dsm1 = np.linalg.norm(v1 - v_sc)
        total_dv += dsm1

        # Flyby at Earth
        v_inf_arr = v2 - v_earth2
        v_inf_mag = np.linalg.norm(v_inf_arr)

        mu_earth = KeplerianEphemeris.MU["earth"]
        r_earth_radius = KeplerianEphemeris.RADIUS["earth"]
        rp = rp_ratio * r_earth_radius

        if v_inf_mag > 1e-6:
            e = 1 + rp * v_inf_mag**2 / mu_earth
            if e > 1:
                delta = 2 * np.arcsin(1 / e)
            else:
                delta = np.pi

            h = np.cross(r_earth2, v_earth2)
            h_norm = np.linalg.norm(h)
            if h_norm > 1e-10:
                h = h / h_norm
            else:
                h = np.array([0, 0, 1])

            cd, sd = np.cos(delta), np.sin(delta)
            v_out = v_inf_arr * cd + np.cross(h, v_inf_arr) * sd + h * np.dot(h, v_inf_arr) * (1 - cd)

            v_hat = v_inf_arr / v_inf_mag
            cb, sb = np.cos(beta), np.sin(beta)
            v_out = v_out * cb + np.cross(v_hat, v_out) * sb

            v_sc = v_earth2 + v_out
        else:
            v_sc = v_earth2

        # Earth-Asteroid leg
        t2 = t1 + tof2
        r_ast, v_ast = self._get_asteroid_position(t2)

        v1, v2, conv = LambertSolver.solve(r_earth2, r_ast, tof2 * DAY, MU_SUN)
        if not conv:
            return 1e10

        # DSM for second leg
        dsm2 = np.linalg.norm(v1 - v_sc)
        total_dv += dsm2

        # Arrival at asteroid (relative velocity for rendezvous)
        v_arr = np.linalg.norm(v2 - v_ast)
        total_dv += v_arr

        return total_dv

    def _create_objective_function(self) -> None:
        """Create the GTOC1 objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._compute_trajectory(x)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        results = xp.zeros(n_points, dtype=X.dtype)

        for i in range(n_points):
            results[i] = self._compute_trajectory(np.asarray(X[i]))

        return results


# =============================================================================
# P16: Rosetta Trajectory
# =============================================================================


class Rosetta(CEC2011Function):
    """P16: Rosetta Spacecraft Trajectory.

    Optimize the Rosetta trajectory to comet 67P/Churyumov-Gerasimenko
    using multiple Earth and Mars gravity assists.

    Sequence: Earth - Earth - Mars - Earth - Earth - Comet (E-E-M-E-E-67P)

    Decision variables (22 total):
    - t0: departure date
    - Vinf, u, v: departure velocity
    - For each leg: TOF, eta (DSM timing)
    - For each flyby: Rp, beta

    Dimension: 22
    Bounds: Problem-specific
    Global optimum: Approximately 1.3 km/s

    References
    ----------
    ESA GTOP Database - Rosetta trajectory.
    """

    _fixed_dim = 22
    _problem_id = 16

    _spec = {
        "problem_id": 16,
        "default_bounds": (0.0, 1.0),
        "continuous": True,
        "differentiable": False,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    _sequence = ["earth", "earth", "mars", "earth", "earth"]

    # Comet 67P orbital elements (simplified)
    _comet_elements = {
        "a": 3.4630,  # AU
        "e": 0.6410,
        "i": 7.0405,  # deg
        "Omega": 50.1420,  # deg
        "omega": 12.7804,  # deg
        "L0": 303.7122,  # deg
        "n": 0.15296,  # deg/day
    }

    _bounds = np.array(
        [
            [1460.0, 1825.0],  # t0: MJD2000 (2004-2005)
            [0.0, 4.0],  # Vinf
            [0.0, 1.0],  # u
            [0.0, 1.0],  # v
            [50.0, 400.0],  # TOF1: E-E
            [1.05, 10.0],  # Rp1
            [-np.pi, np.pi],  # beta1
            [0.0, 1.0],  # eta1
            [100.0, 500.0],  # TOF2: E-M
            [1.05, 10.0],  # Rp2
            [-np.pi, np.pi],  # beta2
            [0.0, 1.0],  # eta2
            [100.0, 700.0],  # TOF3: M-E
            [1.05, 10.0],  # Rp3
            [-np.pi, np.pi],  # beta3
            [0.0, 1.0],  # eta3
            [300.0, 800.0],  # TOF4: E-E
            [1.05, 10.0],  # Rp4
            [-np.pi, np.pi],  # beta4
            [0.0, 1.0],  # eta4
            [800.0, 2500.0],  # TOF5: E-Comet
            [0.0, 1.0],  # eta5
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
        self._f_global = 1.3  # Best known delta-V
        self._x_global = None

    def _get_comet_position(self, mjd2000: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get comet position and velocity."""
        elem = self._comet_elements

        L = elem["L0"] + elem["n"] * mjd2000
        L = np.deg2rad(L % 360.0)

        omega = np.deg2rad(elem["omega"])
        M = L - omega

        e = elem["e"]
        E = KeplerianEphemeris._solve_kepler(M, e)

        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

        a = elem["a"] * AU
        r_mag = a * (1 - e * np.cos(E))

        i = np.deg2rad(elem["i"])
        Omega = np.deg2rad(elem["Omega"])
        omega_arg = omega - Omega

        cos_nu, sin_nu = np.cos(nu), np.sin(nu)
        cos_O, sin_O = np.cos(Omega), np.sin(Omega)
        cos_i, sin_i = np.cos(i), np.sin(i)
        cos_w, sin_w = np.cos(omega_arg), np.sin(omega_arg)

        x_orb = r_mag * cos_nu
        y_orb = r_mag * sin_nu

        r = np.array(
            [
                (cos_O * cos_w - sin_O * sin_w * cos_i) * x_orb + (-cos_O * sin_w - sin_O * cos_w * cos_i) * y_orb,
                (sin_O * cos_w + cos_O * sin_w * cos_i) * x_orb + (-sin_O * sin_w + cos_O * cos_w * cos_i) * y_orb,
                (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb,
            ]
        )

        p = a * (1 - e**2)
        h = np.sqrt(MU_SUN * p)
        vx_orb = -MU_SUN / h * np.sin(nu)
        vy_orb = MU_SUN / h * (e + np.cos(nu))

        v = np.array(
            [
                (cos_O * cos_w - sin_O * sin_w * cos_i) * vx_orb + (-cos_O * sin_w - sin_O * cos_w * cos_i) * vy_orb,
                (sin_O * cos_w + cos_O * sin_w * cos_i) * vx_orb + (-sin_O * sin_w + cos_O * cos_w * cos_i) * vy_orb,
                (sin_w * sin_i) * vx_orb + (cos_w * sin_i) * vy_orb,
            ]
        )

        return r, v

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Convert normalized [0,1] variables to actual bounds."""
        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]
        return lb + x * (ub - lb)

    def _compute_trajectory(self, x: np.ndarray) -> float:
        """Compute delta-V for Rosetta trajectory."""
        x_actual = self._denormalize(x)

        t0 = x_actual[0]
        vinf = x_actual[1]
        u, v = x_actual[2], x_actual[3]

        # Departure direction
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        vinf_dir = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])

        total_dv = 0.0
        current_time = t0

        r_dep, v_dep = KeplerianEphemeris.position_velocity("earth", t0)
        v_sc = v_dep + vinf * vinf_dir

        # Index mapping for decision variables
        idx_tof = [4, 8, 12, 16, 20]
        idx_rp = [5, 9, 13, 17]
        idx_beta = [6, 10, 14, 18]

        r_current = r_dep
        v_current = v_sc
        n_legs = 5

        for leg in range(n_legs):
            tof = x_actual[idx_tof[leg]]
            arrival_time = current_time + tof

            # Get target position
            if leg < 4:
                target = self._sequence[leg + 1]
                r_target, v_target = KeplerianEphemeris.position_velocity(target, arrival_time)
            else:
                r_target, v_target = self._get_comet_position(arrival_time)

            # Solve Lambert
            v1, v2, conv = LambertSolver.solve(r_current, r_target, tof * DAY, MU_SUN)
            if not conv:
                return 1e10

            # DSM
            dsm = np.linalg.norm(v1 - v_current)
            total_dv += dsm

            # Arrival velocity
            v_inf_arr = v2 - v_target
            v_inf_mag = np.linalg.norm(v_inf_arr)

            if leg < n_legs - 1:
                # Flyby
                if leg < 4:
                    target = self._sequence[leg + 1]
                    rp_ratio = x_actual[idx_rp[leg]]
                    beta = x_actual[idx_beta[leg]]

                    mu_p = KeplerianEphemeris.MU.get(target, 1e5)
                    r_p = KeplerianEphemeris.RADIUS.get(target, 6000)
                    rp = rp_ratio * r_p

                    if v_inf_mag > 1e-6:
                        e = 1 + rp * v_inf_mag**2 / mu_p
                        if e > 1:
                            delta = 2 * np.arcsin(1 / e)
                        else:
                            delta = np.pi

                        h = np.cross(r_target, v_target)
                        h_norm = np.linalg.norm(h)
                        if h_norm > 1e-10:
                            h = h / h_norm
                        else:
                            h = np.array([0, 0, 1])

                        cd, sd = np.cos(delta), np.sin(delta)
                        v_out = v_inf_arr * cd + np.cross(h, v_inf_arr) * sd + h * np.dot(h, v_inf_arr) * (1 - cd)

                        v_hat = v_inf_arr / v_inf_mag
                        cb, sb = np.cos(beta), np.sin(beta)
                        v_out = v_out * cb + np.cross(v_hat, v_out) * sb

                        v_current = v_target + v_out
                    else:
                        v_current = v_target
            else:
                # Final arrival at comet
                total_dv += v_inf_mag

            r_current = r_target
            current_time = arrival_time

        return total_dv

    def _create_objective_function(self) -> None:
        """Create the Rosetta objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._compute_trajectory(x)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        results = xp.zeros(n_points, dtype=X.dtype)

        for i in range(n_points):
            results[i] = self._compute_trajectory(np.asarray(X[i]))

        return results


# =============================================================================
# P17: Sagas Solar Sail Trajectory
# =============================================================================


class Sagas(CEC2011Function):
    """P17: Sagas Solar Sail Trajectory.

    Solar sail trajectory optimization to escape the solar system
    using solar radiation pressure for propulsion.

    The solar sail uses continuous low-thrust acceleration from photon
    pressure. The acceleration depends on sail orientation and distance
    from the Sun.

    Decision variables (12 total):
    - t0: departure date
    - sail_area: characteristic sail area
    - Control angles at discrete time points

    Dimension: 12
    Bounds: Problem-specific
    Global optimum: Maximize escape velocity

    Note: This uses a simplified solar sail model with discrete control.

    References
    ----------
    ESA GTOP Database - Sagas trajectory.
    """

    _fixed_dim = 12
    _problem_id = 17

    _spec = {
        "problem_id": 17,
        "default_bounds": (0.0, 1.0),
        "continuous": True,
        "differentiable": False,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    # Solar sail parameters
    _sail_lightness = 0.04  # Characteristic acceleration at 1 AU [mm/s^2]
    _mission_duration = 1000.0  # days

    _bounds = np.array(
        [
            [0.0, 7300.0],  # t0: MJD2000
            [0.0, np.pi / 2],  # alpha0: initial cone angle
            [0.0, 2 * np.pi],  # delta0: initial clock angle
            [0.0, np.pi / 2],  # alpha1
            [0.0, 2 * np.pi],  # delta1
            [0.0, np.pi / 2],  # alpha2
            [0.0, 2 * np.pi],  # delta2
            [0.0, np.pi / 2],  # alpha3
            [0.0, 2 * np.pi],  # delta3
            [0.0, np.pi / 2],  # alpha4
            [0.0, 2 * np.pi],  # delta4
            [100.0, 2000.0],  # mission duration [days]
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
        # Minimize negative of escape velocity (maximize escape velocity)
        self._f_global = -50.0  # Target: high escape velocity
        self._x_global = None

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Convert normalized [0,1] variables to actual bounds."""
        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]
        return lb + x * (ub - lb)

    def _compute_sail_acceleration(
        self, r: np.ndarray, alpha: float, delta: float
    ) -> np.ndarray:
        """Compute solar sail acceleration.

        Parameters
        ----------
        r : np.ndarray
            Position vector [km]
        alpha : float
            Cone angle (angle between sail normal and sun line)
        delta : float
            Clock angle (rotation of sail normal around sun line)

        Returns
        -------
        np.ndarray
            Acceleration vector [km/s^2]
        """
        r_mag = np.linalg.norm(r)
        r_hat = r / r_mag

        # Characteristic acceleration scales with 1/r^2
        a0 = self._sail_lightness * 1e-6  # Convert mm/s^2 to km/s^2
        a_char = a0 * (AU / r_mag) ** 2

        # Sail normal direction in RTN frame
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        # Simplified: acceleration along radial direction modified by cone angle
        # Full model would include transverse components
        a_mag = a_char * cos_a * cos_a  # Acceleration magnitude

        # Direction: combination of radial and transverse
        n_hat = r_hat * cos_a

        # Add transverse component (simplified)
        t_hat = np.array([-r_hat[1], r_hat[0], 0])
        t_norm = np.linalg.norm(t_hat)
        if t_norm > 1e-10:
            t_hat = t_hat / t_norm
        else:
            t_hat = np.array([0, 1, 0])

        # Out-of-plane component
        h_hat = np.cross(r_hat, t_hat)

        # Acceleration direction
        a_dir = cos_a * r_hat + sin_a * (cos_d * t_hat + sin_d * h_hat)
        a_dir = a_dir / np.linalg.norm(a_dir)

        return a_mag * a_dir

    def _propagate_sail(self, x_actual: np.ndarray) -> float:
        """Propagate solar sail trajectory and compute final energy.

        Uses simple Euler integration with piecewise constant control.
        """
        t0 = x_actual[0]
        duration = x_actual[11]

        # Control angles at 5 discrete points
        alphas = [x_actual[1], x_actual[3], x_actual[5], x_actual[7], x_actual[9]]
        deltas = [x_actual[2], x_actual[4], x_actual[6], x_actual[8], x_actual[10]]

        # Initial state: Earth position and velocity
        r, v = KeplerianEphemeris.position_velocity("earth", t0)

        # Integration parameters
        dt = 1.0 * DAY  # 1 day time step
        n_steps = int(duration)
        segment_length = n_steps // 5

        for step in range(n_steps):
            # Determine which control segment we're in
            segment = min(step // max(segment_length, 1), 4)
            alpha = alphas[segment]
            delta = deltas[segment]

            # Gravity
            r_mag = np.linalg.norm(r)
            if r_mag < 1e6:  # Collision with Sun
                return 1e10

            a_grav = -MU_SUN / r_mag**3 * r

            # Solar sail acceleration
            a_sail = self._compute_sail_acceleration(r, alpha, delta)

            # Total acceleration
            a_total = a_grav + a_sail

            # Euler integration
            v = v + a_total * dt
            r = r + v * dt

        # Compute escape velocity (C3 energy)
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        energy = 0.5 * v_mag**2 - MU_SUN / r_mag

        # Return negative energy (we want to maximize escape energy)
        # Positive energy means escape trajectory
        if energy > 0:
            v_inf = np.sqrt(2 * energy)
            return -v_inf  # Negative because we minimize
        else:
            return -energy  # Penalty for bound orbit

    def _compute_trajectory(self, x: np.ndarray) -> float:
        """Compute objective for solar sail trajectory."""
        x_actual = self._denormalize(x)

        try:
            result = self._propagate_sail(x_actual)
        except Exception:
            result = 1e10

        return result

    def _create_objective_function(self) -> None:
        """Create the Sagas objective function."""

        def objective_function(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._compute_trajectory(x)

        self.pure_objective_function = objective_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        results = xp.zeros(n_points, dtype=X.dtype)

        for i in range(n_points):
            results[i] = self._compute_trajectory(np.asarray(X[i]))

        return results


# =============================================================================
# P18-P22: Tight Bounds Variants
# =============================================================================


class Cassini2Tight(Cassini2):
    """P18: Cassini 2 with Tight Bounds.

    Same as P12 (Cassini 2) but with tighter variable bounds,
    making the problem easier by reducing the search space.

    Dimension: 22
    Global optimum: Same as P12 (~8.4 km/s)
    """

    _problem_id = 18

    _spec = {
        "problem_id": 18,
        "default_bounds": (0.0, 1.0),
        "continuous": True,
        "differentiable": False,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    # Tighter bounds around known good solutions
    _bounds = np.array(
        [
            [-800.0, -600.0],  # t0: narrower window
            [3.5, 4.5],  # Vinf
            [0.3, 0.7],  # u
            [0.3, 0.7],  # v
            [150.0, 300.0],  # TOF1
            [1.1, 5.0],  # Rp1
            [-np.pi / 2, np.pi / 2],  # beta1
            [150.0, 400.0],  # TOF2
            [1.1, 5.0],  # Rp2
            [-np.pi / 2, np.pi / 2],  # beta2
            [50.0, 200.0],  # TOF3
            [1.1, 5.0],  # Rp3
            [-np.pi / 2, np.pi / 2],  # beta3
            [600.0, 1200.0],  # TOF4
            [1.1, 50.0],  # Rp4
            [-np.pi / 2, np.pi / 2],  # beta4
            [1000.0, 1800.0],  # TOF5
            [0.2, 0.8],  # eta1
            [0.2, 0.8],  # eta2
            [0.2, 0.8],  # eta3
            [0.2, 0.8],  # eta4
            [0.2, 0.8],  # eta5
        ]
    )


class MessengerTight(Messenger):
    """P19: Messenger with Tight Bounds.

    Same as P13 (Messenger) but with tighter variable bounds,
    making the problem easier by reducing the search space.

    Dimension: 26
    Global optimum: Same as P13 (~8.6 km/s)
    """

    _problem_id = 19

    _spec = {
        "problem_id": 19,
        "default_bounds": (0.0, 1.0),
        "continuous": True,
        "differentiable": False,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    # Tighter bounds
    _bounds = np.array(
        [
            [2000.0, 2200.0],  # t0
            [3.0, 3.8],  # Vinf
            [0.3, 0.7],  # u
            [0.3, 0.7],  # v
            [150.0, 400.0],  # TOF1
            [1.1, 5.0],  # Rp1
            [-np.pi / 2, np.pi / 2],  # beta1
            [100.0, 300.0],  # TOF2
            [1.1, 5.0],  # Rp2
            [-np.pi / 2, np.pi / 2],  # beta2
            [80.0, 200.0],  # TOF3
            [1.1, 5.0],  # Rp3
            [-np.pi / 2, np.pi / 2],  # beta3
            [80.0, 200.0],  # TOF4
            [1.1, 5.0],  # Rp4
            [-np.pi / 2, np.pi / 2],  # beta4
            [80.0, 200.0],  # TOF5
            [1.1, 5.0],  # Rp5
            [-np.pi / 2, np.pi / 2],  # beta5
            [80.0, 200.0],  # TOF6
            [0.2, 0.8],  # eta1
            [0.2, 0.8],  # eta2
            [0.2, 0.8],  # eta3
            [0.2, 0.8],  # eta4
            [0.2, 0.8],  # eta5
            [0.2, 0.8],  # eta6
        ]
    )


class GTOC1Tight(GTOC1):
    """P20: GTOC1 with Tight Bounds.

    Same as P15 (GTOC1) but with tighter variable bounds.

    Dimension: 8
    Global optimum: Same as P15
    """

    _problem_id = 20

    _spec = {
        "problem_id": 20,
        "default_bounds": (0.0, 1.0),
        "continuous": True,
        "differentiable": False,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    _bounds = np.array(
        [
            [5000.0, 8000.0],  # t0
            [300.0, 1200.0],  # TOF1
            [300.0, 1200.0],  # TOF2
            [1.0, 3.0],  # Vinf
            [0.3, 0.7],  # u
            [0.3, 0.7],  # v
            [1.1, 5.0],  # Rp
            [-np.pi / 2, np.pi / 2],  # beta
        ]
    )


class RosettaTight(Rosetta):
    """P21: Rosetta with Tight Bounds.

    Same as P16 (Rosetta) but with tighter variable bounds.

    Dimension: 22
    Global optimum: Same as P16 (~1.3 km/s)
    """

    _problem_id = 21

    _spec = {
        "problem_id": 21,
        "default_bounds": (0.0, 1.0),
        "continuous": True,
        "differentiable": False,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    _bounds = np.array(
        [
            [1550.0, 1700.0],  # t0
            [0.5, 2.5],  # Vinf
            [0.3, 0.7],  # u
            [0.3, 0.7],  # v
            [100.0, 300.0],  # TOF1
            [1.1, 5.0],  # Rp1
            [-np.pi / 2, np.pi / 2],  # beta1
            [0.2, 0.8],  # eta1
            [200.0, 400.0],  # TOF2
            [1.1, 5.0],  # Rp2
            [-np.pi / 2, np.pi / 2],  # beta2
            [0.2, 0.8],  # eta2
            [200.0, 500.0],  # TOF3
            [1.1, 5.0],  # Rp3
            [-np.pi / 2, np.pi / 2],  # beta3
            [0.2, 0.8],  # eta3
            [400.0, 700.0],  # TOF4
            [1.1, 5.0],  # Rp4
            [-np.pi / 2, np.pi / 2],  # beta4
            [0.2, 0.8],  # eta4
            [1200.0, 2000.0],  # TOF5
            [0.2, 0.8],  # eta5
        ]
    )


class SagasTight(Sagas):
    """P22: Sagas with Tight Bounds.

    Same as P17 (Sagas) but with tighter variable bounds.

    Dimension: 12
    Global optimum: Same as P17
    """

    _problem_id = 22

    _spec = {
        "problem_id": 22,
        "default_bounds": (0.0, 1.0),
        "continuous": True,
        "differentiable": False,
        "scalable": False,
        "unimodal": False,
        "separable": False,
    }

    _bounds = np.array(
        [
            [1000.0, 5000.0],  # t0
            [np.pi / 6, np.pi / 3],  # alpha0
            [0.0, np.pi],  # delta0
            [np.pi / 6, np.pi / 3],  # alpha1
            [0.0, np.pi],  # delta1
            [np.pi / 6, np.pi / 3],  # alpha2
            [0.0, np.pi],  # delta2
            [np.pi / 6, np.pi / 3],  # alpha3
            [0.0, np.pi],  # delta3
            [np.pi / 6, np.pi / 3],  # alpha4
            [0.0, np.pi],  # delta4
            [500.0, 1500.0],  # duration
        ]
    )


# =============================================================================
# Collections
# =============================================================================

CEC2011_SPACECRAFT: List[type] = [
    Cassini2,
    Messenger,
    Cassini1,
    GTOC1,
    Rosetta,
    Sagas,
    Cassini2Tight,
    MessengerTight,
    GTOC1Tight,
    RosettaTight,
    SagasTight,
]
