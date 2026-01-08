# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2021 Hybrid Benchmark Functions (F5-F7).

These functions are identical to CEC 2020 but use different shift/rotation data.
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2021 import CEC2021Function


class HybridFunction2021Base(CEC2021Function):
    """Base class for CEC 2021 hybrid functions."""

    _proportions: Tuple[float, ...] = ()

    def _get_group_sizes(self) -> List[int]:
        D = self.n_dim
        sizes = []
        remaining = D
        for i, p in enumerate(self._proportions[:-1]):
            size = max(1, int(round(p * D)))
            sizes.append(min(size, remaining - (len(self._proportions) - i - 1)))
            remaining -= sizes[-1]
        sizes.append(remaining)
        return sizes

    def _shuffle_and_split(self, z: np.ndarray) -> List[np.ndarray]:
        shuffle_idx = self._get_shuffle_indices()
        z_shuffled = z[shuffle_idx]
        sizes = self._get_group_sizes()
        groups = []
        start = 0
        for size in sizes:
            groups.append(z_shuffled[start : start + size])
            start += size
        return groups


class HybridFunction1_2021(HybridFunction2021Base):
    """F5: Hybrid Function 1 (Schwefel + Rastrigin + Elliptic)."""

    _spec = {
        "name": "Hybrid Function 1",
        "func_id": 5,
        "unimodal": False,
        "separable": False,
    }
    _proportions = (0.3, 0.3, 0.4)

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

    def _elliptic(self, z: np.ndarray) -> float:
        D = len(z)
        if D == 1:
            return z[0] ** 2
        coeffs = np.power(1e6, np.arange(D) / (D - 1))
        return np.sum(coeffs * z**2)

    def _create_objective_function(self) -> None:
        def hybrid1(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            groups = self._shuffle_and_split(z)
            result = self._schwefel(groups[0])
            result += self._rastrigin(groups[1])
            result += self._elliptic(groups[2])
            return result + self.f_global

        self.pure_objective_function = hybrid1

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        shuffle_idx = self._get_shuffle_indices()
        Z_shuffled = Z[:, shuffle_idx]
        sizes = self._get_group_sizes()
        start = 0
        result = xp.zeros(X.shape[0], dtype=X.dtype)

        # Schwefel
        g1 = Z_shuffled[:, start : start + sizes[0]]
        g1_scaled = 10.0 * g1 + 420.9687462275036
        D1 = sizes[0]
        case1 = xp.abs(g1_scaled) <= 500
        term1 = g1_scaled * xp.sin(xp.sqrt(xp.abs(g1_scaled)))
        case2 = g1_scaled > 500
        zm2 = 500 - g1_scaled % 500
        term2 = zm2 * xp.sin(xp.sqrt(xp.abs(zm2))) - (g1_scaled - 500) ** 2 / (10000 * D1)
        zm3 = xp.abs(g1_scaled) % 500 - 500
        term3 = zm3 * xp.sin(xp.sqrt(xp.abs(zm3))) - (g1_scaled + 500) ** 2 / (10000 * D1)
        contrib = xp.where(case1, term1, xp.where(case2, term2, term3))
        result += 418.9829 * D1 - xp.sum(contrib, axis=1)
        start += sizes[0]

        # Rastrigin
        g2 = Z_shuffled[:, start : start + sizes[1]]
        g2_scaled = 0.0512 * g2
        D2 = sizes[1]
        result += 10 * D2 + xp.sum(g2_scaled**2 - 10 * xp.cos(2 * math.pi * g2_scaled), axis=1)
        start += sizes[1]

        # Elliptic
        g3 = Z_shuffled[:, start : start + sizes[2]]
        D3 = sizes[2]
        if D3 > 1:
            coeffs = xp.asarray(np.power(1e6, np.arange(D3) / (D3 - 1)))
            result += xp.sum(coeffs * g3**2, axis=1)
        else:
            result += g3[:, 0] ** 2

        return result + self.f_global


class HybridFunction2_2021(HybridFunction2021Base):
    """F6: Hybrid Function 2 (Schaffer F6 + HGBat + Rosenbrock + Schwefel)."""

    _spec = {
        "name": "Hybrid Function 2",
        "func_id": 6,
        "unimodal": False,
        "separable": False,
    }
    _proportions = (0.2, 0.2, 0.3, 0.3)

    def _expanded_schaffer_f6(self, z: np.ndarray) -> float:
        D = len(z)
        result = 0.0
        for i in range(D - 1):
            t = z[i] ** 2 + z[i + 1] ** 2
            result += 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        t = z[-1] ** 2 + z[0] ** 2
        result += 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        return result

    def _hgbat(self, z: np.ndarray) -> float:
        D = len(z)
        z = 0.05 * z
        sum_z = np.sum(z)
        sum_z2 = np.sum(z**2)
        return np.sqrt(abs(sum_z2**2 - sum_z**2)) + (0.5 * sum_z2 + sum_z) / D + 0.5

    def _rosenbrock(self, z: np.ndarray) -> float:
        z = 0.02048 * z + 1.0
        result = 0.0
        for i in range(len(z) - 1):
            result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
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

    def _create_objective_function(self) -> None:
        def hybrid2(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            groups = self._shuffle_and_split(z)
            result = self._expanded_schaffer_f6(groups[0])
            result += self._hgbat(groups[1])
            result += self._rosenbrock(groups[2])
            result += self._schwefel(groups[3])
            return result + self.f_global

        self.pure_objective_function = hybrid2

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        shuffle_idx = self._get_shuffle_indices()
        Z_shuffled = Z[:, shuffle_idx]
        sizes = self._get_group_sizes()
        start = 0
        result = xp.zeros(X.shape[0], dtype=X.dtype)

        # Expanded Schaffer's F6
        g1 = Z_shuffled[:, start : start + sizes[0]]
        D1 = sizes[0]
        for i in range(D1 - 1):
            t = g1[:, i] ** 2 + g1[:, i + 1] ** 2
            result += 0.5 + (xp.sin(xp.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        if D1 > 0:
            t = g1[:, -1] ** 2 + g1[:, 0] ** 2
            result += 0.5 + (xp.sin(xp.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        start += sizes[0]

        # HGBat
        g2 = Z_shuffled[:, start : start + sizes[1]]
        g2_scaled = 0.05 * g2
        D2 = sizes[1]
        sum_z = xp.sum(g2_scaled, axis=1)
        sum_z2 = xp.sum(g2_scaled**2, axis=1)
        result += xp.sqrt(xp.abs(sum_z2**2 - sum_z**2)) + (0.5 * sum_z2 + sum_z) / D2 + 0.5
        start += sizes[1]

        # Rosenbrock
        g3 = Z_shuffled[:, start : start + sizes[2]]
        g3_scaled = 0.02048 * g3 + 1.0
        D3 = sizes[2]
        if D3 > 1:
            result += xp.sum(
                100 * (g3_scaled[:, :-1] ** 2 - g3_scaled[:, 1:]) ** 2 + (g3_scaled[:, :-1] - 1) ** 2, axis=1
            )
        start += sizes[2]

        # Schwefel
        g4 = Z_shuffled[:, start : start + sizes[3]]
        g4_scaled = 10.0 * g4 + 420.9687462275036
        D4 = sizes[3]
        case1 = xp.abs(g4_scaled) <= 500
        term1 = g4_scaled * xp.sin(xp.sqrt(xp.abs(g4_scaled)))
        case2 = g4_scaled > 500
        zm2 = 500 - g4_scaled % 500
        term2 = zm2 * xp.sin(xp.sqrt(xp.abs(zm2))) - (g4_scaled - 500) ** 2 / (10000 * D4)
        zm3 = xp.abs(g4_scaled) % 500 - 500
        term3 = zm3 * xp.sin(xp.sqrt(xp.abs(zm3))) - (g4_scaled + 500) ** 2 / (10000 * D4)
        contrib = xp.where(case1, term1, xp.where(case2, term2, term3))
        result += 418.9829 * D4 - xp.sum(contrib, axis=1)

        return result + self.f_global


class HybridFunction3_2021(HybridFunction2021Base):
    """F7: Hybrid Function 3 (Schaffer F6 + HGBat + Rosenbrock + Schwefel + Elliptic)."""

    _spec = {
        "name": "Hybrid Function 3",
        "func_id": 7,
        "unimodal": False,
        "separable": False,
    }
    _proportions = (0.1, 0.2, 0.2, 0.2, 0.3)

    def _expanded_schaffer_f6(self, z: np.ndarray) -> float:
        D = len(z)
        result = 0.0
        for i in range(D - 1):
            t = z[i] ** 2 + z[i + 1] ** 2
            result += 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        t = z[-1] ** 2 + z[0] ** 2
        result += 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        return result

    def _hgbat(self, z: np.ndarray) -> float:
        D = len(z)
        z = 0.05 * z
        sum_z = np.sum(z)
        sum_z2 = np.sum(z**2)
        return np.sqrt(abs(sum_z2**2 - sum_z**2)) + (0.5 * sum_z2 + sum_z) / D + 0.5

    def _rosenbrock(self, z: np.ndarray) -> float:
        z = 0.02048 * z + 1.0
        result = 0.0
        for i in range(len(z) - 1):
            result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
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

    def _elliptic(self, z: np.ndarray) -> float:
        D = len(z)
        if D == 1:
            return z[0] ** 2
        coeffs = np.power(1e6, np.arange(D) / (D - 1))
        return np.sum(coeffs * z**2)

    def _create_objective_function(self) -> None:
        def hybrid3(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            groups = self._shuffle_and_split(z)
            result = self._expanded_schaffer_f6(groups[0])
            result += self._hgbat(groups[1])
            result += self._rosenbrock(groups[2])
            result += self._schwefel(groups[3])
            result += self._elliptic(groups[4])
            return result + self.f_global

        self.pure_objective_function = hybrid3

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        shuffle_idx = self._get_shuffle_indices()
        Z_shuffled = Z[:, shuffle_idx]
        sizes = self._get_group_sizes()
        start = 0
        result = xp.zeros(X.shape[0], dtype=X.dtype)

        # Expanded Schaffer's F6
        g1 = Z_shuffled[:, start : start + sizes[0]]
        D1 = sizes[0]
        for i in range(D1 - 1):
            t = g1[:, i] ** 2 + g1[:, i + 1] ** 2
            result += 0.5 + (xp.sin(xp.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        if D1 > 0:
            t = g1[:, -1] ** 2 + g1[:, 0] ** 2
            result += 0.5 + (xp.sin(xp.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
        start += sizes[0]

        # HGBat
        g2 = Z_shuffled[:, start : start + sizes[1]]
        g2_scaled = 0.05 * g2
        D2 = sizes[1]
        sum_z = xp.sum(g2_scaled, axis=1)
        sum_z2 = xp.sum(g2_scaled**2, axis=1)
        result += xp.sqrt(xp.abs(sum_z2**2 - sum_z**2)) + (0.5 * sum_z2 + sum_z) / D2 + 0.5
        start += sizes[1]

        # Rosenbrock
        g3 = Z_shuffled[:, start : start + sizes[2]]
        g3_scaled = 0.02048 * g3 + 1.0
        D3 = sizes[2]
        if D3 > 1:
            result += xp.sum(
                100 * (g3_scaled[:, :-1] ** 2 - g3_scaled[:, 1:]) ** 2 + (g3_scaled[:, :-1] - 1) ** 2, axis=1
            )
        start += sizes[2]

        # Schwefel
        g4 = Z_shuffled[:, start : start + sizes[3]]
        g4_scaled = 10.0 * g4 + 420.9687462275036
        D4 = sizes[3]
        case1 = xp.abs(g4_scaled) <= 500
        term1 = g4_scaled * xp.sin(xp.sqrt(xp.abs(g4_scaled)))
        case2 = g4_scaled > 500
        zm2 = 500 - g4_scaled % 500
        term2 = zm2 * xp.sin(xp.sqrt(xp.abs(zm2))) - (g4_scaled - 500) ** 2 / (10000 * D4)
        zm3 = xp.abs(g4_scaled) % 500 - 500
        term3 = zm3 * xp.sin(xp.sqrt(xp.abs(zm3))) - (g4_scaled + 500) ** 2 / (10000 * D4)
        contrib = xp.where(case1, term1, xp.where(case2, term2, term3))
        result += 418.9829 * D4 - xp.sum(contrib, axis=1)
        start += sizes[3]

        # Elliptic
        g5 = Z_shuffled[:, start : start + sizes[4]]
        D5 = sizes[4]
        if D5 > 1:
            coeffs = xp.asarray(np.power(1e6, np.arange(D5) / (D5 - 1)))
            result += xp.sum(coeffs * g5**2, axis=1)
        else:
            result += g5[:, 0] ** 2

        return result + self.f_global


CEC2021_HYBRID = [
    HybridFunction1_2021,
    HybridFunction2_2021,
    HybridFunction3_2021,
]
