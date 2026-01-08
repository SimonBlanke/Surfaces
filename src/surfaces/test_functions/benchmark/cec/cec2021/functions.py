# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2021 Basic Benchmark Functions (F1-F4).

These functions are identical to CEC 2020 but use different shift/rotation data.
"""

import math
from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2021 import CEC2021Function


class ShiftedRotatedBentCigar2021(CEC2021Function):
    """F1: Shifted and Rotated Bent Cigar Function."""

    _spec = {
        "name": "Shifted and Rotated Bent Cigar Function",
        "func_id": 1,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def bent_cigar(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2) + self.f_global

        self.pure_objective_function = bent_cigar

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        return Z[:, 0] ** 2 + 1e6 * xp.sum(Z[:, 1:] ** 2, axis=1) + self.f_global


class ShiftedRotatedSchwefel2021(CEC2021Function):
    """F2: Shifted and Rotated Schwefel's Function."""

    _spec = {
        "name": "Shifted and Rotated Schwefel's Function",
        "func_id": 2,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def schwefel(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = 10.0 * z + 420.9687462275036

            D = self.n_dim
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

            return 418.9829 * D - result + self.f_global

        self.pure_objective_function = schwefel

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        D = self.n_dim
        Z = self._batch_shift_rotate(X)
        Z = 10.0 * Z + 420.9687462275036

        case1 = xp.abs(Z) <= 500
        term1 = Z * xp.sin(xp.sqrt(xp.abs(Z)))
        case2 = Z > 500
        zm2 = 500 - Z % 500
        term2 = zm2 * xp.sin(xp.sqrt(xp.abs(zm2))) - (Z - 500) ** 2 / (10000 * D)
        zm3 = xp.abs(Z) % 500 - 500
        term3 = zm3 * xp.sin(xp.sqrt(xp.abs(zm3))) - (Z + 500) ** 2 / (10000 * D)
        contrib = xp.where(case1, term1, xp.where(case2, term2, term3))
        result = 418.9829 * D - xp.sum(contrib, axis=1)
        return result + self.f_global


class ShiftedRotatedLunacekBiRastrigin2021(CEC2021Function):
    """F3: Shifted and Rotated Lunacek Bi-Rastrigin's Function."""

    _spec = {
        "name": "Shifted and Rotated Lunacek Bi-Rastrigin's Function",
        "func_id": 3,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def lunacek_bi_rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            shift = self._get_shift_vector()
            M = self._get_rotation_matrix()

            D = self.n_dim
            mu0 = 2.5
            s = 1 - 1 / (2 * np.sqrt(D + 20) - 8.2)
            mu1 = -np.sqrt((mu0**2 - 1) / s)

            y = 0.1 * (x - shift)
            tmpx = 2 * y.copy()
            tmpx[shift < 0] *= -1
            z = tmpx.copy()
            tmpx = tmpx + mu0

            t1 = np.sum((tmpx - mu0) ** 2)
            t2 = s * np.sum((tmpx - mu1) ** 2) + D
            y = M @ z
            t = np.sum(np.cos(2 * np.pi * y))
            result = min(t1, t2) + 10 * (D - t)
            return result + self.f_global

        self.pure_objective_function = lunacek_bi_rastrigin

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        D = self.n_dim

        shift = xp.asarray(self._get_shift_vector())
        M = xp.asarray(self._get_rotation_matrix())

        mu0 = 2.5
        s = 1 - 1 / (2 * math.sqrt(D + 20) - 8.2)
        mu1 = -math.sqrt((mu0**2 - 1) / s)

        Y = 0.1 * (X - shift)
        TMPX = 2 * Y
        sign_mask = xp.where(shift < 0, -1.0, 1.0)
        TMPX = TMPX * sign_mask
        Z = TMPX.copy()
        TMPX = TMPX + mu0

        t1 = xp.sum((TMPX - mu0) ** 2, axis=1)
        t2 = s * xp.sum((TMPX - mu1) ** 2, axis=1) + D
        Y_rot = Z @ M.T
        t = xp.sum(xp.cos(2 * math.pi * Y_rot), axis=1)
        result = xp.minimum(t1, t2) + 10 * (D - t)
        return result + self.f_global


class ExpandedGriewankRosenbrock2021(CEC2021Function):
    """F4: Expanded Rosenbrock's plus Griewank's Function."""

    _spec = {
        "name": "Expanded Rosenbrock's plus Griewank's Function",
        "func_id": 4,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def expanded_griewank_rosenbrock(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = 0.05 * z + 1.0

            D = self.n_dim
            result = 0.0
            for i in range(D - 1):
                t = 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
                result += t**2 / 4000 - np.cos(t) + 1
            t = 100 * (z[-1] ** 2 - z[0]) ** 2 + (z[-1] - 1) ** 2
            result += t**2 / 4000 - np.cos(t) + 1
            return result + self.f_global

        self.pure_objective_function = expanded_griewank_rosenbrock

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        Z = 0.05 * Z + 1.0

        z_curr = Z[:, :-1]
        z_next = Z[:, 1:]
        t = 100 * (z_curr**2 - z_next) ** 2 + (z_curr - 1) ** 2
        result = xp.sum(t**2 / 4000 - xp.cos(t) + 1, axis=1)
        t_wrap = 100 * (Z[:, -1] ** 2 - Z[:, 0]) ** 2 + (Z[:, -1] - 1) ** 2
        result += t_wrap**2 / 4000 - xp.cos(t_wrap) + 1
        return result + self.f_global


CEC2021_BASIC = [
    ShiftedRotatedBentCigar2021,
    ShiftedRotatedSchwefel2021,
    ShiftedRotatedLunacekBiRastrigin2021,
    ExpandedGriewankRosenbrock2021,
]
