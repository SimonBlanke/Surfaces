# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2019 100-Digit Challenge benchmark functions F1-F10.

These functions are from the CEC 2019 Special Session on 100-Digit Challenge
on Single Objective Numerical Optimization.

References
----------
Price, K. V., Awad, N. H., Ali, M. Z., & Suganthan, P. N. (2018).
Problem definitions and evaluation criteria for the 100-Digit Challenge
special session and competition on single objective numerical optimization.
"""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2019 import CEC2019Function


# =============================================================================
# F1-F3: Special Functions (different dimensions)
# =============================================================================


class StornsChebyshev(CEC2019Function):
    """F1: Storn's Chebyshev Polynomial Fitting Problem.

    A polynomial fitting problem with D=9 dimensions.
    """

    _spec = {
        "name": "Storn's Chebyshev Polynomial Fitting",
        "func_id": 1,
        "default_bounds": (-8192.0, 8192.0),
        "unimodal": False,
        "separable": False,
    }

    _fixed_dim = 9

    def _create_objective_function(self) -> None:
        def chebyshev(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)

            # Target Chebyshev polynomial coefficients
            n = len(x)
            m = 32 * n  # Number of sample points

            result = 0.0
            for i in range(m + 1):
                t = -1 + 2 * i / m
                # Compute polynomial value
                p = x[0]
                for j in range(1, n):
                    p = p * t + x[j]

                # Target Chebyshev value
                if abs(t) <= 1:
                    target = np.cos(n * np.arccos(t))
                else:
                    target = np.cosh(n * np.arccosh(abs(t)))

                result += (p - target) ** 2

            return result + self.f_global

        self.pure_objective_function = chebyshev

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n_batch = X.shape[0]
        n = X.shape[1]
        m = 32 * n

        results = xp.zeros(n_batch)
        for k in range(n_batch):
            x = X[k]
            result = 0.0
            for i in range(m + 1):
                t = -1 + 2 * i / m
                p = x[0]
                for j in range(1, n):
                    p = p * t + x[j]

                if abs(t) <= 1:
                    target = np.cos(n * np.arccos(t))
                else:
                    target = np.cosh(n * np.arccosh(abs(t)))

                result += (p - target) ** 2
            results[k] = result

        return results + self.f_global


class InverseHilbert(CEC2019Function):
    """F2: Inverse Hilbert Matrix Problem.

    A matrix inversion problem with D=16 dimensions.
    """

    _spec = {
        "name": "Inverse Hilbert Matrix Problem",
        "func_id": 2,
        "default_bounds": (-16384.0, 16384.0),
        "unimodal": True,
        "separable": False,
    }

    _fixed_dim = 16

    def _create_objective_function(self) -> None:
        def inverse_hilbert(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            n = int(np.sqrt(len(x)))

            # Reshape to matrix
            X = x.reshape(n, n)

            # Create Hilbert matrix
            H = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    H[i, j] = 1.0 / (i + j + 1)

            # Compute ||X*H - I||_F
            product = X @ H
            identity = np.eye(n)
            result = np.sum((product - identity) ** 2)

            return result + self.f_global

        self.pure_objective_function = inverse_hilbert

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n_batch = X.shape[0]
        d = X.shape[1]
        n = int(np.sqrt(d))

        # Hilbert matrix
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = 1.0 / (i + j + 1)
        H = xp.asarray(H)
        I = xp.eye(n)

        results = xp.zeros(n_batch)
        for k in range(n_batch):
            mat = X[k].reshape(n, n)
            product = mat @ H
            results[k] = xp.sum((product - I) ** 2)

        return results + self.f_global


class LennardJones(CEC2019Function):
    """F3: Lennard-Jones Minimum Energy Cluster Problem.

    Find atomic configuration with minimum Lennard-Jones potential energy.
    D=18 (6 atoms * 3 coordinates).
    """

    _spec = {
        "name": "Lennard-Jones Minimum Energy Cluster",
        "func_id": 3,
        "default_bounds": (-4.0, 4.0),
        "unimodal": False,
        "separable": False,
    }

    _fixed_dim = 18

    def _create_objective_function(self) -> None:
        def lennard_jones(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            n_atoms = len(x) // 3

            # Reshape to atom coordinates
            coords = x.reshape(n_atoms, 3)

            # Compute pairwise Lennard-Jones potential
            energy = 0.0
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    r2 = np.sum((coords[i] - coords[j]) ** 2)
                    if r2 < 1e-10:
                        r2 = 1e-10
                    r6 = r2 ** 3
                    r12 = r6 ** 2
                    energy += 1.0 / r12 - 2.0 / r6

            # Shift to have minimum at f* = 1
            # Known minimum for 6 atoms is approximately -12.712
            return energy + 12.712 + self.f_global

        self.pure_objective_function = lennard_jones

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n_batch = X.shape[0]
        n_atoms = X.shape[1] // 3

        results = xp.zeros(n_batch)
        for k in range(n_batch):
            coords = X[k].reshape(n_atoms, 3)
            energy = 0.0
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    r2 = xp.sum((coords[i] - coords[j]) ** 2)
                    if r2 < 1e-10:
                        r2 = 1e-10
                    r6 = r2 ** 3
                    r12 = r6 ** 2
                    energy += 1.0 / r12 - 2.0 / r6
            results[k] = energy + 12.712

        return results + self.f_global


# =============================================================================
# F4-F10: Shifted and Rotated Functions (D=10)
# =============================================================================


class ShiftedRotatedRastrigin2019(CEC2019Function):
    """F4: Shifted and Rotated Rastrigin's Function."""

    _spec = {
        "name": "Shifted and Rotated Rastrigin's Function",
        "func_id": 4,
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
        "separable": False,
    }

    _fixed_dim = 10

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            return np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10) + self.f_global
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        return xp.sum(Z**2 - 10 * xp.cos(2 * np.pi * Z) + 10, axis=1) + self.f_global


class ShiftedRotatedGriewank2019(CEC2019Function):
    """F5: Shifted and Rotated Griewank's Function."""

    _spec = {
        "name": "Shifted and Rotated Griewank's Function",
        "func_id": 5,
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
        "separable": False,
    }

    _fixed_dim = 10

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            D = len(z)
            i = np.arange(1, D + 1)
            return np.sum(z**2) / 4000 - np.prod(np.cos(z / np.sqrt(i))) + 1 + self.f_global
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        D = Z.shape[1]
        i = xp.arange(1, D + 1, dtype=X.dtype)
        return xp.sum(Z**2, axis=1) / 4000 - xp.prod(xp.cos(Z / xp.sqrt(i)), axis=1) + 1 + self.f_global


class ShiftedRotatedWeierstrass2019(CEC2019Function):
    """F6: Shifted and Rotated Weierstrass Function."""

    _spec = {
        "name": "Shifted and Rotated Weierstrass Function",
        "func_id": 6,
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
        "separable": False,
    }

    _fixed_dim = 10

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x) * 0.5 / 100

            a, b, k_max = 0.5, 3.0, 20
            D = len(z)
            result = 0.0
            for i in range(D):
                for k in range(k_max + 1):
                    result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))

            const = D * sum(a**k * np.cos(np.pi * b**k) for k in range(k_max + 1))
            return result - const + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X) * 0.5 / 100

        a, b, k_max = 0.5, 3.0, 20
        D = Z.shape[1]
        n = X.shape[0]

        result = xp.zeros(n)
        for k in range(k_max + 1):
            result = result + a**k * xp.sum(xp.cos(2 * np.pi * b**k * (Z + 0.5)), axis=1)

        const = D * sum(a**kk * np.cos(np.pi * b**kk) for kk in range(k_max + 1))
        return result - const + self.f_global


class ShiftedRotatedSchwefel2019(CEC2019Function):
    """F7: Shifted and Rotated Schwefel's Function."""

    _spec = {
        "name": "Shifted and Rotated Schwefel's Function",
        "func_id": 7,
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
        "separable": False,
    }

    _fixed_dim = 10

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x) + 4.209687462275036e2

            D = len(z)
            result = 0.0
            for i in range(D):
                zi = z[i]
                if abs(zi) <= 500:
                    result += zi * np.sin(np.sqrt(abs(zi)))
                elif zi > 500:
                    result += (500 - zi % 500) * np.sin(np.sqrt(abs(500 - zi % 500)))
                    result -= (zi - 500)**2 / (10000 * D)
                else:
                    result += (abs(zi) % 500 - 500) * np.sin(np.sqrt(abs(abs(zi) % 500 - 500)))
                    result -= (zi + 500)**2 / (10000 * D)

            return 418.9829 * D - result + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X) + 4.209687462275036e2

        D = Z.shape[1]
        n = X.shape[0]

        result = xp.zeros(n)
        for i in range(D):
            zi = Z[:, i]
            mask_normal = xp.abs(zi) <= 500
            mask_high = zi > 500
            mask_low = zi < -500

            result = result + xp.where(mask_normal, zi * xp.sin(xp.sqrt(xp.abs(zi))), 0.0)

            tmp_high = 500 - zi % 500
            result = result + xp.where(mask_high,
                tmp_high * xp.sin(xp.sqrt(xp.abs(tmp_high))) - (zi - 500)**2 / (10000 * D), 0.0)

            tmp_low = xp.abs(zi) % 500 - 500
            result = result + xp.where(mask_low,
                tmp_low * xp.sin(xp.sqrt(xp.abs(tmp_low))) - (zi + 500)**2 / (10000 * D), 0.0)

        return 418.9829 * D - result + self.f_global


class ExpandedScafferF62019(CEC2019Function):
    """F8: Shifted and Rotated Expanded Schaffer's F6 Function."""

    _spec = {
        "name": "Shifted and Rotated Expanded Schaffer's F6",
        "func_id": 8,
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
        "separable": False,
    }

    _fixed_dim = 10

    def _create_objective_function(self) -> None:
        def scaffer_f6(x, y):
            tmp = x**2 + y**2
            return 0.5 + (np.sin(np.sqrt(tmp))**2 - 0.5) / (1 + 0.001 * tmp)**2

        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            D = len(z)
            result = 0.0
            for i in range(D):
                result += scaffer_f6(z[i], z[(i + 1) % D])

            return result + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)

        D = Z.shape[1]
        n = X.shape[0]

        result = xp.zeros(n)
        for i in range(D):
            j = (i + 1) % D
            tmp = Z[:, i]**2 + Z[:, j]**2
            result = result + 0.5 + (xp.sin(xp.sqrt(tmp))**2 - 0.5) / (1 + 0.001 * tmp)**2

        return result + self.f_global


class ShiftedRotatedHappyCat2019(CEC2019Function):
    """F9: Shifted and Rotated Happy Cat Function."""

    _spec = {
        "name": "Shifted and Rotated Happy Cat Function",
        "func_id": 9,
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
        "separable": False,
    }

    _fixed_dim = 10

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x) * 5 / 100

            D = len(z)
            sum_z = np.sum(z)
            sum_z2 = np.sum(z**2)

            return abs(sum_z2 - D)**0.25 + (0.5 * sum_z2 + sum_z) / D + 0.5 + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X) * 5 / 100

        D = Z.shape[1]
        sum_z = xp.sum(Z, axis=1)
        sum_z2 = xp.sum(Z**2, axis=1)

        return xp.abs(sum_z2 - D)**0.25 + (0.5 * sum_z2 + sum_z) / D + 0.5 + self.f_global


class ShiftedRotatedAckley2019(CEC2019Function):
    """F10: Shifted and Rotated Ackley's Function."""

    _spec = {
        "name": "Shifted and Rotated Ackley's Function",
        "func_id": 10,
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
        "separable": False,
    }

    _fixed_dim = 10

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            D = len(z)
            term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / D))
            term2 = -np.exp(np.sum(np.cos(2 * np.pi * z)) / D)

            return term1 + term2 + 20 + np.e + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)

        D = Z.shape[1]
        term1 = -20 * xp.exp(-0.2 * xp.sqrt(xp.sum(Z**2, axis=1) / D))
        term2 = -xp.exp(xp.sum(xp.cos(2 * np.pi * Z), axis=1) / D)

        return term1 + term2 + 20 + np.e + self.f_global


# =============================================================================
# All CEC 2019 functions
# =============================================================================

CEC2019_ALL = [
    StornsChebyshev,
    InverseHilbert,
    LennardJones,
    ShiftedRotatedRastrigin2019,
    ShiftedRotatedGriewank2019,
    ShiftedRotatedWeierstrass2019,
    ShiftedRotatedSchwefel2019,
    ExpandedScafferF62019,
    ShiftedRotatedHappyCat2019,
    ShiftedRotatedAckley2019,
]

CEC2019_SPECIAL = [
    StornsChebyshev,
    InverseHilbert,
    LennardJones,
]

CEC2019_STANDARD = [
    ShiftedRotatedRastrigin2019,
    ShiftedRotatedGriewank2019,
    ShiftedRotatedWeierstrass2019,
    ShiftedRotatedSchwefel2019,
    ExpandedScafferF62019,
    ShiftedRotatedHappyCat2019,
    ShiftedRotatedAckley2019,
]
