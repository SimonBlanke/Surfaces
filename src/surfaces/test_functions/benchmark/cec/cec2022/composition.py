# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2022 Composition Benchmark Functions (F9-F12)."""

import math
from typing import Any, Dict, Tuple

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2022 import CEC2022Function


class CompositionFunction2022Base(CEC2022Function):
    """Base class for CEC 2022 composition functions."""

    _sigmas: Tuple[float, ...] = ()
    _lambdas: Tuple[float, ...] = ()
    _num_funcs: int = 0

    def _compute_weights(self, x: np.ndarray) -> np.ndarray:
        weights = np.zeros(self._num_funcs)
        for i in range(self._num_funcs):
            shift = self._get_shift_vector(i + 1)
            diff = x - shift
            weights[i] = np.exp(-np.sum(diff**2) / (2 * self.n_dim * self._sigmas[i] ** 2))

        max_w = np.max(weights)
        for i in range(self._num_funcs):
            if weights[i] != max_w:
                weights[i] *= 1 - max_w**10

        weights /= np.sum(weights) + 1e-10
        return weights

    def _batch_compute_weights(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n_points = X.shape[0]

        weights = xp.zeros((n_points, self._num_funcs), dtype=X.dtype)
        for i in range(self._num_funcs):
            shift = xp.asarray(self._get_shift_vector(i + 1))
            diff = X - shift
            weights[:, i] = xp.exp(-xp.sum(diff**2, axis=1) / (2 * self.n_dim * self._sigmas[i] ** 2))

        max_w = xp.max(weights, axis=1, keepdims=True)
        mask = weights != max_w
        weights = xp.where(mask, weights * (1 - max_w**10), weights)
        weights = weights / (xp.sum(weights, axis=1, keepdims=True) + 1e-10)
        return weights


class CompositionFunction1_2022(CompositionFunction2022Base):
    """F9: Composition Function 1.

    Components:
    - Rosenbrock, Elliptic, Bent Cigar, Discus
    """

    _spec = {
        "name": "Composition Function 1",
        "func_id": 9,
        "unimodal": False,
        "separable": False,
    }
    _sigmas = (10.0, 20.0, 30.0, 40.0)
    _lambdas = (1.0, 1e-6, 1.0, 1.0)
    _num_funcs = 4

    def _rosenbrock(self, z: np.ndarray) -> float:
        z = 0.02048 * z + 1.0
        result = 0.0
        for i in range(len(z) - 1):
            result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
        return result

    def _elliptic(self, z: np.ndarray) -> float:
        D = len(z)
        if D == 1:
            return z[0] ** 2
        coeffs = np.power(1e6, np.arange(D) / (D - 1))
        return np.sum(coeffs * z**2)

    def _bent_cigar(self, z: np.ndarray) -> float:
        return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2)

    def _discus(self, z: np.ndarray) -> float:
        return 1e6 * z[0] ** 2 + np.sum(z[1:] ** 2)

    def _create_objective_function(self) -> None:
        def composition1(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            weights = self._compute_weights(x)

            funcs = [self._rosenbrock, self._elliptic, self._bent_cigar, self._discus]
            result = 0.0
            for i, func in enumerate(funcs):
                z = self._rotate(x - self._get_shift_vector(i + 1), i + 1)
                result += weights[i] * self._lambdas[i] * func(z)

            return result + self.f_global

        self.pure_objective_function = composition1

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        D = self.n_dim

        weights = self._batch_compute_weights(X)
        result = xp.zeros(X.shape[0], dtype=X.dtype)

        # Rosenbrock
        shift1 = xp.asarray(self._get_shift_vector(1))
        M1 = xp.asarray(self._get_rotation_matrix(1))
        Z1 = (X - shift1) @ M1.T
        Z1_scaled = 0.02048 * Z1 + 1.0
        f1 = self._lambdas[0] * xp.sum(
            100 * (Z1_scaled[:, :-1] ** 2 - Z1_scaled[:, 1:]) ** 2 + (Z1_scaled[:, :-1] - 1) ** 2, axis=1
        )
        result += weights[:, 0] * f1

        # Elliptic
        shift2 = xp.asarray(self._get_shift_vector(2))
        M2 = xp.asarray(self._get_rotation_matrix(2))
        Z2 = (X - shift2) @ M2.T
        if D > 1:
            coeffs = xp.asarray(np.power(1e6, np.arange(D) / (D - 1)))
        else:
            coeffs = xp.asarray([1.0])
        f2 = self._lambdas[1] * xp.sum(coeffs * Z2**2, axis=1)
        result += weights[:, 1] * f2

        # Bent Cigar
        shift3 = xp.asarray(self._get_shift_vector(3))
        M3 = xp.asarray(self._get_rotation_matrix(3))
        Z3 = (X - shift3) @ M3.T
        f3 = self._lambdas[2] * (Z3[:, 0] ** 2 + 1e6 * xp.sum(Z3[:, 1:] ** 2, axis=1))
        result += weights[:, 2] * f3

        # Discus
        shift4 = xp.asarray(self._get_shift_vector(4))
        M4 = xp.asarray(self._get_rotation_matrix(4))
        Z4 = (X - shift4) @ M4.T
        f4 = self._lambdas[3] * (1e6 * Z4[:, 0] ** 2 + xp.sum(Z4[:, 1:] ** 2, axis=1))
        result += weights[:, 3] * f4

        return result + self.f_global


class CompositionFunction2_2022(CompositionFunction2022Base):
    """F10: Composition Function 2.

    Components:
    - Schwefel, Rastrigin, HGBat
    """

    _spec = {
        "name": "Composition Function 2",
        "func_id": 10,
        "unimodal": False,
        "separable": False,
    }
    _sigmas = (10.0, 20.0, 30.0)
    _lambdas = (1.0, 10.0, 1.0)
    _num_funcs = 3

    def _schwefel(self, z: np.ndarray) -> float:
        D = len(z)
        z = 10.0 * z + 420.9687462275036
        result = 0.0
        for i in range(D):
            zi = z[i]
            if abs(zi) <= 500:
                result += zi * np.sin(np.sqrt(abs(zi)))
            elif zi > 500:
                zm = 500 - zi % 500
                result += zm * np.sin(np.sqrt(abs(zm)))
                result -= (zi - 500) ** 2 / (10000 * D)
            else:
                zm = abs(zi) % 500 - 500
                result += zm * np.sin(np.sqrt(abs(zm)))
                result -= (zi + 500) ** 2 / (10000 * D)
        return 418.9829 * D - result

    def _rastrigin(self, z: np.ndarray) -> float:
        D = len(z)
        z = 0.0512 * z
        return 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

    def _hgbat(self, z: np.ndarray) -> float:
        D = len(z)
        z = 0.05 * z
        sum_z = np.sum(z)
        sum_z2 = np.sum(z**2)
        return np.sqrt(abs(sum_z2**2 - sum_z**2)) + (0.5 * sum_z2 + sum_z) / D + 0.5

    def _create_objective_function(self) -> None:
        def composition2(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            weights = self._compute_weights(x)

            funcs = [self._schwefel, self._rastrigin, self._hgbat]
            result = 0.0
            for i, func in enumerate(funcs):
                z = self._rotate(x - self._get_shift_vector(i + 1), i + 1)
                result += weights[i] * self._lambdas[i] * func(z)

            return result + self.f_global

        self.pure_objective_function = composition2

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        D = self.n_dim

        weights = self._batch_compute_weights(X)
        result = xp.zeros(X.shape[0], dtype=X.dtype)

        # Schwefel
        shift1 = xp.asarray(self._get_shift_vector(1))
        M1 = xp.asarray(self._get_rotation_matrix(1))
        Z1 = (X - shift1) @ M1.T
        Z1_scaled = 10.0 * Z1 + 420.9687462275036
        case1 = xp.abs(Z1_scaled) <= 500
        term1 = Z1_scaled * xp.sin(xp.sqrt(xp.abs(Z1_scaled)))
        case2 = Z1_scaled > 500
        zm2 = 500 - Z1_scaled % 500
        term2 = zm2 * xp.sin(xp.sqrt(xp.abs(zm2))) - (Z1_scaled - 500) ** 2 / (10000 * D)
        zm3 = xp.abs(Z1_scaled) % 500 - 500
        term3 = zm3 * xp.sin(xp.sqrt(xp.abs(zm3))) - (Z1_scaled + 500) ** 2 / (10000 * D)
        contrib = xp.where(case1, term1, xp.where(case2, term2, term3))
        f1 = self._lambdas[0] * (418.9829 * D - xp.sum(contrib, axis=1))
        result += weights[:, 0] * f1

        # Rastrigin
        shift2 = xp.asarray(self._get_shift_vector(2))
        M2 = xp.asarray(self._get_rotation_matrix(2))
        Z2 = (X - shift2) @ M2.T
        Z2_scaled = 0.0512 * Z2
        f2 = self._lambdas[1] * (10 * D + xp.sum(Z2_scaled**2 - 10 * xp.cos(2 * math.pi * Z2_scaled), axis=1))
        result += weights[:, 1] * f2

        # HGBat
        shift3 = xp.asarray(self._get_shift_vector(3))
        M3 = xp.asarray(self._get_rotation_matrix(3))
        Z3 = (X - shift3) @ M3.T
        Z3_scaled = 0.05 * Z3
        sum_z = xp.sum(Z3_scaled, axis=1)
        sum_z2 = xp.sum(Z3_scaled**2, axis=1)
        f3 = self._lambdas[2] * (xp.sqrt(xp.abs(sum_z2**2 - sum_z**2)) + (0.5 * sum_z2 + sum_z) / D + 0.5)
        result += weights[:, 2] * f3

        return result + self.f_global


class CompositionFunction3_2022(CompositionFunction2022Base):
    """F11: Composition Function 3.

    Components:
    - Expanded Schaffer F6, Schwefel, Griewank, Rosenbrock, Rastrigin
    """

    _spec = {
        "name": "Composition Function 3",
        "func_id": 11,
        "unimodal": False,
        "separable": False,
    }
    _sigmas = (10.0, 20.0, 30.0, 40.0, 50.0)
    _lambdas = (10.0, 1.0, 10.0, 1.0, 1.0)
    _num_funcs = 5

    def _expanded_schaffer_f6(self, z: np.ndarray) -> float:
        D = len(z)
        result = 0.0
        for i in range(D - 1):
            t = z[i] ** 2 + z[i + 1] ** 2
            result += 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        t = z[-1] ** 2 + z[0] ** 2
        result += 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        return result

    def _schwefel(self, z: np.ndarray) -> float:
        D = len(z)
        z = 10.0 * z + 420.9687462275036
        result = 0.0
        for i in range(D):
            zi = z[i]
            if abs(zi) <= 500:
                result += zi * np.sin(np.sqrt(abs(zi)))
            elif zi > 500:
                zm = 500 - zi % 500
                result += zm * np.sin(np.sqrt(abs(zm)))
                result -= (zi - 500) ** 2 / (10000 * D)
            else:
                zm = abs(zi) % 500 - 500
                result += zm * np.sin(np.sqrt(abs(zm)))
                result -= (zi + 500) ** 2 / (10000 * D)
        return 418.9829 * D - result

    def _griewank(self, z: np.ndarray) -> float:
        D = len(z)
        z = 6.0 * z
        sum_sq = np.sum(z**2)
        prod_cos = np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1))))
        return sum_sq / 4000 - prod_cos + 1

    def _rosenbrock(self, z: np.ndarray) -> float:
        z = 0.02048 * z + 1.0
        result = 0.0
        for i in range(len(z) - 1):
            result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
        return result

    def _rastrigin(self, z: np.ndarray) -> float:
        D = len(z)
        z = 0.0512 * z
        return 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

    def _create_objective_function(self) -> None:
        def composition3(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            weights = self._compute_weights(x)

            funcs = [self._expanded_schaffer_f6, self._schwefel, self._griewank, self._rosenbrock, self._rastrigin]
            result = 0.0
            for i, func in enumerate(funcs):
                z = self._rotate(x - self._get_shift_vector(i + 1), i + 1)
                result += weights[i] * self._lambdas[i] * func(z)

            return result + self.f_global

        self.pure_objective_function = composition3

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        D = self.n_dim

        weights = self._batch_compute_weights(X)
        result = xp.zeros(X.shape[0], dtype=X.dtype)

        # Expanded Schaffer F6
        shift1 = xp.asarray(self._get_shift_vector(1))
        M1 = xp.asarray(self._get_rotation_matrix(1))
        Z1 = (X - shift1) @ M1.T
        f1_val = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(D - 1):
            t = Z1[:, i] ** 2 + Z1[:, i + 1] ** 2
            f1_val += 0.5 + (xp.sin(xp.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        t = Z1[:, -1] ** 2 + Z1[:, 0] ** 2
        f1_val += 0.5 + (xp.sin(xp.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        result += weights[:, 0] * self._lambdas[0] * f1_val

        # Schwefel
        shift2 = xp.asarray(self._get_shift_vector(2))
        M2 = xp.asarray(self._get_rotation_matrix(2))
        Z2 = (X - shift2) @ M2.T
        Z2_scaled = 10.0 * Z2 + 420.9687462275036
        case1 = xp.abs(Z2_scaled) <= 500
        term1 = Z2_scaled * xp.sin(xp.sqrt(xp.abs(Z2_scaled)))
        case2 = Z2_scaled > 500
        zm2 = 500 - Z2_scaled % 500
        term2 = zm2 * xp.sin(xp.sqrt(xp.abs(zm2))) - (Z2_scaled - 500) ** 2 / (10000 * D)
        zm3 = xp.abs(Z2_scaled) % 500 - 500
        term3 = zm3 * xp.sin(xp.sqrt(xp.abs(zm3))) - (Z2_scaled + 500) ** 2 / (10000 * D)
        contrib = xp.where(case1, term1, xp.where(case2, term2, term3))
        f2 = self._lambdas[1] * (418.9829 * D - xp.sum(contrib, axis=1))
        result += weights[:, 1] * f2

        # Griewank
        shift3 = xp.asarray(self._get_shift_vector(3))
        M3 = xp.asarray(self._get_rotation_matrix(3))
        Z3 = (X - shift3) @ M3.T
        Z3_scaled = 6.0 * Z3
        sum_sq = xp.sum(Z3_scaled**2, axis=1)
        indices = xp.asarray(np.sqrt(np.arange(1, D + 1)))
        prod_cos = xp.prod(xp.cos(Z3_scaled / indices), axis=1)
        f3 = self._lambdas[2] * (sum_sq / 4000 - prod_cos + 1)
        result += weights[:, 2] * f3

        # Rosenbrock
        shift4 = xp.asarray(self._get_shift_vector(4))
        M4 = xp.asarray(self._get_rotation_matrix(4))
        Z4 = (X - shift4) @ M4.T
        Z4_scaled = 0.02048 * Z4 + 1.0
        f4 = self._lambdas[3] * xp.sum(
            100 * (Z4_scaled[:, :-1] ** 2 - Z4_scaled[:, 1:]) ** 2 + (Z4_scaled[:, :-1] - 1) ** 2, axis=1
        )
        result += weights[:, 3] * f4

        # Rastrigin
        shift5 = xp.asarray(self._get_shift_vector(5))
        M5 = xp.asarray(self._get_rotation_matrix(5))
        Z5 = (X - shift5) @ M5.T
        Z5_scaled = 0.0512 * Z5
        f5 = self._lambdas[4] * (10 * D + xp.sum(Z5_scaled**2 - 10 * xp.cos(2 * math.pi * Z5_scaled), axis=1))
        result += weights[:, 4] * f5

        return result + self.f_global


class CompositionFunction4_2022(CompositionFunction2022Base):
    """F12: Composition Function 4.

    Components:
    - HGBat, Rastrigin, Schwefel, Bent Cigar, Elliptic, Expanded Schaffer F6
    """

    _spec = {
        "name": "Composition Function 4",
        "func_id": 12,
        "unimodal": False,
        "separable": False,
    }
    _sigmas = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    _lambdas = (10.0, 10.0, 2.5, 1.0, 1e-6, 10.0)
    _num_funcs = 6

    def _hgbat(self, z: np.ndarray) -> float:
        D = len(z)
        z = 0.05 * z
        sum_z = np.sum(z)
        sum_z2 = np.sum(z**2)
        return np.sqrt(abs(sum_z2**2 - sum_z**2)) + (0.5 * sum_z2 + sum_z) / D + 0.5

    def _rastrigin(self, z: np.ndarray) -> float:
        D = len(z)
        z = 0.0512 * z
        return 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

    def _schwefel(self, z: np.ndarray) -> float:
        D = len(z)
        z = 10.0 * z + 420.9687462275036
        result = 0.0
        for i in range(D):
            zi = z[i]
            if abs(zi) <= 500:
                result += zi * np.sin(np.sqrt(abs(zi)))
            elif zi > 500:
                zm = 500 - zi % 500
                result += zm * np.sin(np.sqrt(abs(zm)))
                result -= (zi - 500) ** 2 / (10000 * D)
            else:
                zm = abs(zi) % 500 - 500
                result += zm * np.sin(np.sqrt(abs(zm)))
                result -= (zi + 500) ** 2 / (10000 * D)
        return 418.9829 * D - result

    def _bent_cigar(self, z: np.ndarray) -> float:
        return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2)

    def _elliptic(self, z: np.ndarray) -> float:
        D = len(z)
        if D == 1:
            return z[0] ** 2
        coeffs = np.power(1e6, np.arange(D) / (D - 1))
        return np.sum(coeffs * z**2)

    def _expanded_schaffer_f6(self, z: np.ndarray) -> float:
        D = len(z)
        result = 0.0
        for i in range(D - 1):
            t = z[i] ** 2 + z[i + 1] ** 2
            result += 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        t = z[-1] ** 2 + z[0] ** 2
        result += 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        return result

    def _create_objective_function(self) -> None:
        def composition4(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            weights = self._compute_weights(x)

            funcs = [
                self._hgbat,
                self._rastrigin,
                self._schwefel,
                self._bent_cigar,
                self._elliptic,
                self._expanded_schaffer_f6,
            ]
            result = 0.0
            for i, func in enumerate(funcs):
                z = self._rotate(x - self._get_shift_vector(i + 1), i + 1)
                result += weights[i] * self._lambdas[i] * func(z)

            return result + self.f_global

        self.pure_objective_function = composition4

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        D = self.n_dim

        weights = self._batch_compute_weights(X)
        result = xp.zeros(X.shape[0], dtype=X.dtype)

        # HGBat
        shift1 = xp.asarray(self._get_shift_vector(1))
        M1 = xp.asarray(self._get_rotation_matrix(1))
        Z1 = (X - shift1) @ M1.T
        Z1_scaled = 0.05 * Z1
        sum_z = xp.sum(Z1_scaled, axis=1)
        sum_z2 = xp.sum(Z1_scaled**2, axis=1)
        f1 = self._lambdas[0] * (xp.sqrt(xp.abs(sum_z2**2 - sum_z**2)) + (0.5 * sum_z2 + sum_z) / D + 0.5)
        result += weights[:, 0] * f1

        # Rastrigin
        shift2 = xp.asarray(self._get_shift_vector(2))
        M2 = xp.asarray(self._get_rotation_matrix(2))
        Z2 = (X - shift2) @ M2.T
        Z2_scaled = 0.0512 * Z2
        f2 = self._lambdas[1] * (10 * D + xp.sum(Z2_scaled**2 - 10 * xp.cos(2 * math.pi * Z2_scaled), axis=1))
        result += weights[:, 1] * f2

        # Schwefel
        shift3 = xp.asarray(self._get_shift_vector(3))
        M3 = xp.asarray(self._get_rotation_matrix(3))
        Z3 = (X - shift3) @ M3.T
        Z3_scaled = 10.0 * Z3 + 420.9687462275036
        case1 = xp.abs(Z3_scaled) <= 500
        term1 = Z3_scaled * xp.sin(xp.sqrt(xp.abs(Z3_scaled)))
        case2 = Z3_scaled > 500
        zm2 = 500 - Z3_scaled % 500
        term2 = zm2 * xp.sin(xp.sqrt(xp.abs(zm2))) - (Z3_scaled - 500) ** 2 / (10000 * D)
        zm3 = xp.abs(Z3_scaled) % 500 - 500
        term3 = zm3 * xp.sin(xp.sqrt(xp.abs(zm3))) - (Z3_scaled + 500) ** 2 / (10000 * D)
        contrib = xp.where(case1, term1, xp.where(case2, term2, term3))
        f3 = self._lambdas[2] * (418.9829 * D - xp.sum(contrib, axis=1))
        result += weights[:, 2] * f3

        # Bent Cigar
        shift4 = xp.asarray(self._get_shift_vector(4))
        M4 = xp.asarray(self._get_rotation_matrix(4))
        Z4 = (X - shift4) @ M4.T
        f4 = self._lambdas[3] * (Z4[:, 0] ** 2 + 1e6 * xp.sum(Z4[:, 1:] ** 2, axis=1))
        result += weights[:, 3] * f4

        # Elliptic
        shift5 = xp.asarray(self._get_shift_vector(5))
        M5 = xp.asarray(self._get_rotation_matrix(5))
        Z5 = (X - shift5) @ M5.T
        if D > 1:
            coeffs = xp.asarray(np.power(1e6, np.arange(D) / (D - 1)))
        else:
            coeffs = xp.asarray([1.0])
        f5 = self._lambdas[4] * xp.sum(coeffs * Z5**2, axis=1)
        result += weights[:, 4] * f5

        # Expanded Schaffer F6
        shift6 = xp.asarray(self._get_shift_vector(6))
        M6 = xp.asarray(self._get_rotation_matrix(6))
        Z6 = (X - shift6) @ M6.T
        f6_val = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(D - 1):
            t = Z6[:, i] ** 2 + Z6[:, i + 1] ** 2
            f6_val += 0.5 + (xp.sin(xp.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        t = Z6[:, -1] ** 2 + Z6[:, 0] ** 2
        f6_val += 0.5 + (xp.sin(xp.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        result += weights[:, 5] * self._lambdas[5] * f6_val

        return result + self.f_global


CEC2022_COMPOSITION = [
    CompositionFunction1_2022,
    CompositionFunction2_2022,
    CompositionFunction3_2022,
    CompositionFunction4_2022,
]
