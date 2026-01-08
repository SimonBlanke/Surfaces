# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2022 Hybrid Benchmark Functions (F6-F8)."""

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2022 import CEC2022Function


class HybridFunction2022Base(CEC2022Function):
    """Base class for CEC 2022 hybrid functions."""

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


class HybridFunction1_2022(HybridFunction2022Base):
    """F6: Hybrid Function 1.

    Components:
    - Bent Cigar (20%)
    - HGBat (20%)
    - Rastrigin (30%)
    """

    _spec = {
        "name": "Hybrid Function 1",
        "func_id": 6,
        "unimodal": False,
        "separable": False,
    }
    _proportions = (0.2, 0.2, 0.3, 0.3)  # 4 components

    def _bent_cigar(self, z: np.ndarray) -> float:
        return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2)

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

    def _create_objective_function(self) -> None:
        def hybrid1(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            groups = self._shuffle_and_split(z)
            result = self._bent_cigar(groups[0])
            result += self._hgbat(groups[1])
            result += self._rastrigin(groups[2])
            result += self._schwefel(groups[3])
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

        # Bent Cigar
        g1 = Z_shuffled[:, start : start + sizes[0]]
        result += g1[:, 0] ** 2 + 1e6 * xp.sum(g1[:, 1:] ** 2, axis=1)
        start += sizes[0]

        # HGBat
        g2 = Z_shuffled[:, start : start + sizes[1]]
        g2_scaled = 0.05 * g2
        D2 = sizes[1]
        sum_z = xp.sum(g2_scaled, axis=1)
        sum_z2 = xp.sum(g2_scaled**2, axis=1)
        result += xp.sqrt(xp.abs(sum_z2**2 - sum_z**2)) + (0.5 * sum_z2 + sum_z) / D2 + 0.5
        start += sizes[1]

        # Rastrigin
        g3 = Z_shuffled[:, start : start + sizes[2]]
        g3_scaled = 0.0512 * g3
        D3 = sizes[2]
        result += 10 * D3 + xp.sum(g3_scaled**2 - 10 * xp.cos(2 * math.pi * g3_scaled), axis=1)
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


class HybridFunction2_2022(HybridFunction2022Base):
    """F7: Hybrid Function 2.

    Components:
    - HGBat, Katsuura, Ackley, Rastrigin, Schwefel, Schaffer F7 (6 components)
    """

    _spec = {
        "name": "Hybrid Function 2",
        "func_id": 7,
        "unimodal": False,
        "separable": False,
    }
    _proportions = (0.1, 0.1, 0.2, 0.2, 0.2, 0.2)

    def _hgbat(self, z: np.ndarray) -> float:
        D = len(z)
        z = 0.05 * z
        sum_z = np.sum(z)
        sum_z2 = np.sum(z**2)
        return np.sqrt(abs(sum_z2**2 - sum_z**2)) + (0.5 * sum_z2 + sum_z) / D + 0.5

    def _katsuura(self, z: np.ndarray) -> float:
        D = len(z)
        z = 0.05 * z
        result = 1.0
        for i in range(D):
            tmp = 0.0
            for j in range(1, 33):
                tmp += abs(2**j * z[i] - round(2**j * z[i])) / (2**j)
            result *= (1 + (i + 1) * tmp) ** (10.0 / D**1.2)
        return 10.0 / D**2 * result - 10.0 / D**2

    def _ackley(self, z: np.ndarray) -> float:
        D = len(z)
        sum_sq = np.sum(z**2)
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        return -20 * np.exp(-0.2 * np.sqrt(sum_sq / D)) - np.exp(sum_cos / D) + 20 + np.e

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

    def _schaffer_f7(self, z: np.ndarray) -> float:
        D = len(z)
        si = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        tmp = np.sin(50 * (si**0.2))
        sm = np.sum(np.sqrt(si) * (tmp**2 + 1))
        return (sm**2) / ((D - 1) ** 2)

    def _create_objective_function(self) -> None:
        def hybrid2(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            groups = self._shuffle_and_split(z)
            result = self._hgbat(groups[0])
            result += self._katsuura(groups[1])
            result += self._ackley(groups[2])
            result += self._rastrigin(groups[3])
            result += self._schwefel(groups[4])
            result += self._schaffer_f7(groups[5])
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

        # HGBat
        g1 = Z_shuffled[:, start : start + sizes[0]]
        g1_scaled = 0.05 * g1
        D1 = sizes[0]
        sum_z = xp.sum(g1_scaled, axis=1)
        sum_z2 = xp.sum(g1_scaled**2, axis=1)
        result += xp.sqrt(xp.abs(sum_z2**2 - sum_z**2)) + (0.5 * sum_z2 + sum_z) / D1 + 0.5
        start += sizes[0]

        # Katsuura (scalar implementation for batch)
        g2 = Z_shuffled[:, start : start + sizes[1]]
        g2_scaled = 0.05 * g2
        D2 = sizes[1]
        for n in range(X.shape[0]):
            prod = 1.0
            for i in range(D2):
                tmp = 0.0
                for j in range(1, 33):
                    tmp += abs(2**j * g2_scaled[n, i] - round(2**j * g2_scaled[n, i])) / (2**j)
                prod *= (1 + (i + 1) * tmp) ** (10.0 / D2**1.2)
            result[n] += 10.0 / D2**2 * prod - 10.0 / D2**2
        start += sizes[1]

        # Ackley
        g3 = Z_shuffled[:, start : start + sizes[2]]
        D3 = sizes[2]
        sum_sq = xp.sum(g3**2, axis=1)
        sum_cos = xp.sum(xp.cos(2 * math.pi * g3), axis=1)
        result += -20 * xp.exp(-0.2 * xp.sqrt(sum_sq / D3)) - xp.exp(sum_cos / D3) + 20 + math.e
        start += sizes[2]

        # Rastrigin
        g4 = Z_shuffled[:, start : start + sizes[3]]
        g4_scaled = 0.0512 * g4
        D4 = sizes[3]
        result += 10 * D4 + xp.sum(g4_scaled**2 - 10 * xp.cos(2 * math.pi * g4_scaled), axis=1)
        start += sizes[3]

        # Schwefel
        g5 = Z_shuffled[:, start : start + sizes[4]]
        g5_scaled = 10.0 * g5 + 420.9687462275036
        D5 = sizes[4]
        case1 = xp.abs(g5_scaled) <= 500
        term1 = g5_scaled * xp.sin(xp.sqrt(xp.abs(g5_scaled)))
        case2 = g5_scaled > 500
        zm2 = 500 - g5_scaled % 500
        term2 = zm2 * xp.sin(xp.sqrt(xp.abs(zm2))) - (g5_scaled - 500) ** 2 / (10000 * D5)
        zm3 = xp.abs(g5_scaled) % 500 - 500
        term3 = zm3 * xp.sin(xp.sqrt(xp.abs(zm3))) - (g5_scaled + 500) ** 2 / (10000 * D5)
        contrib = xp.where(case1, term1, xp.where(case2, term2, term3))
        result += 418.9829 * D5 - xp.sum(contrib, axis=1)
        start += sizes[4]

        # Schaffer F7
        g6 = Z_shuffled[:, start : start + sizes[5]]
        D6 = sizes[5]
        if D6 > 1:
            S = xp.sqrt(g6[:, :-1] ** 2 + g6[:, 1:] ** 2)
            tmp = xp.sin(50 * (S**0.2))
            sm = xp.sum(xp.sqrt(S) * (tmp**2 + 1), axis=1)
            result += (sm**2) / ((D6 - 1) ** 2)

        return result + self.f_global


class HybridFunction3_2022(HybridFunction2022Base):
    """F8: Hybrid Function 3.

    Components:
    - Katsuura, HappyCat, Griewank-Rosenbrock, Schwefel, Ackley (5 components)
    """

    _spec = {
        "name": "Hybrid Function 3",
        "func_id": 8,
        "unimodal": False,
        "separable": False,
    }
    _proportions = (0.1, 0.2, 0.2, 0.2, 0.3)

    def _katsuura(self, z: np.ndarray) -> float:
        D = len(z)
        z = 0.05 * z
        result = 1.0
        for i in range(D):
            tmp = 0.0
            for j in range(1, 33):
                tmp += abs(2**j * z[i] - round(2**j * z[i])) / (2**j)
            result *= (1 + (i + 1) * tmp) ** (10.0 / D**1.2)
        return 10.0 / D**2 * result - 10.0 / D**2

    def _happy_cat(self, z: np.ndarray) -> float:
        D = len(z)
        z = 0.05 * z
        sum_z = np.sum(z)
        sum_z2 = np.sum(z**2)
        return abs(sum_z2 - D) ** 0.25 + (0.5 * sum_z2 + sum_z) / D + 0.5

    def _griewank_rosenbrock(self, z: np.ndarray) -> float:
        z = 0.05 * z + 1.0
        D = len(z)
        result = 0.0
        for i in range(D - 1):
            t = 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
            result += t**2 / 4000 - np.cos(t) + 1
        t = 100 * (z[-1] ** 2 - z[0]) ** 2 + (z[-1] - 1) ** 2
        result += t**2 / 4000 - np.cos(t) + 1
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

    def _ackley(self, z: np.ndarray) -> float:
        D = len(z)
        sum_sq = np.sum(z**2)
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        return -20 * np.exp(-0.2 * np.sqrt(sum_sq / D)) - np.exp(sum_cos / D) + 20 + np.e

    def _create_objective_function(self) -> None:
        def hybrid3(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            groups = self._shuffle_and_split(z)
            result = self._katsuura(groups[0])
            result += self._happy_cat(groups[1])
            result += self._griewank_rosenbrock(groups[2])
            result += self._schwefel(groups[3])
            result += self._ackley(groups[4])
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

        # Katsuura (scalar for batch)
        g1 = Z_shuffled[:, start : start + sizes[0]]
        g1_scaled = 0.05 * g1
        D1 = sizes[0]
        for n in range(X.shape[0]):
            prod = 1.0
            for i in range(D1):
                tmp = 0.0
                for j in range(1, 33):
                    tmp += abs(2**j * g1_scaled[n, i] - round(2**j * g1_scaled[n, i])) / (2**j)
                prod *= (1 + (i + 1) * tmp) ** (10.0 / D1**1.2)
            result[n] += 10.0 / D1**2 * prod - 10.0 / D1**2
        start += sizes[0]

        # HappyCat
        g2 = Z_shuffled[:, start : start + sizes[1]]
        g2_scaled = 0.05 * g2
        D2 = sizes[1]
        sum_z = xp.sum(g2_scaled, axis=1)
        sum_z2 = xp.sum(g2_scaled**2, axis=1)
        result += xp.abs(sum_z2 - D2) ** 0.25 + (0.5 * sum_z2 + sum_z) / D2 + 0.5
        start += sizes[1]

        # Griewank-Rosenbrock
        g3 = Z_shuffled[:, start : start + sizes[2]]
        g3_scaled = 0.05 * g3 + 1.0
        D3 = sizes[2]
        if D3 > 1:
            z_curr = g3_scaled[:, :-1]
            z_next = g3_scaled[:, 1:]
            t = 100 * (z_curr**2 - z_next) ** 2 + (z_curr - 1) ** 2
            result += xp.sum(t**2 / 4000 - xp.cos(t) + 1, axis=1)
        t_wrap = 100 * (g3_scaled[:, -1] ** 2 - g3_scaled[:, 0]) ** 2 + (g3_scaled[:, -1] - 1) ** 2
        result += t_wrap**2 / 4000 - xp.cos(t_wrap) + 1
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

        # Ackley
        g5 = Z_shuffled[:, start : start + sizes[4]]
        D5 = sizes[4]
        sum_sq = xp.sum(g5**2, axis=1)
        sum_cos = xp.sum(xp.cos(2 * math.pi * g5), axis=1)
        result += -20 * xp.exp(-0.2 * xp.sqrt(sum_sq / D5)) - xp.exp(sum_cos / D5) + 20 + math.e

        return result + self.f_global


CEC2022_HYBRID = [
    HybridFunction1_2022,
    HybridFunction2_2022,
    HybridFunction3_2022,
]
