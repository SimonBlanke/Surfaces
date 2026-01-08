# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2022 Basic Benchmark Functions (F1-F5)."""

import math
from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2022 import CEC2022Function


class ShiftedRotatedZakharov2022(CEC2022Function):
    """F1: Shifted and Full Rotated Zakharov Function.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Full Rotated Zakharov Function",
        "func_id": 1,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def zakharov(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            D = self.n_dim
            sum1 = np.sum(z**2)
            sum2 = np.sum(0.5 * np.arange(1, D + 1) * z)
            return sum1 + sum2**2 + sum2**4 + self.f_global

        self.pure_objective_function = zakharov

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        D = self.n_dim
        Z = self._batch_shift_rotate(X)
        i = xp.arange(1, D + 1, dtype=X.dtype)
        sum1 = xp.sum(Z**2, axis=1)
        sum2 = xp.sum(0.5 * i * Z, axis=1)
        return sum1 + sum2**2 + sum2**4 + self.f_global


class ShiftedRotatedRosenbrock2022(CEC2022Function):
    """F2: Shifted and Rotated Rosenbrock's Function.

    Properties:
    - Multimodal (huge number of local optima)
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Rosenbrock's Function",
        "func_id": 2,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def rosenbrock(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = 0.02048 * z + 1.0

            result = 0.0
            for i in range(self.n_dim - 1):
                result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2

            return result + self.f_global

        self.pure_objective_function = rosenbrock

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        Z = 0.02048 * Z + 1.0
        result = xp.sum(100 * (Z[:, :-1] ** 2 - Z[:, 1:]) ** 2 + (Z[:, :-1] - 1) ** 2, axis=1)
        return result + self.f_global


class ShiftedRotatedExpandedSchafferF72022(CEC2022Function):
    """F3: Shifted and Full Rotated Expanded Schaffer's F7 Function.

    Properties:
    - Multimodal (asymmetrical)
    - Non-separable
    - Huge number of local optima
    """

    _spec = {
        "name": "Shifted and Full Rotated Expanded Schaffer's F7 Function",
        "func_id": 3,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def expanded_schaffer_f7(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            D = self.n_dim
            si = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
            tmp = np.sin(50 * (si**0.2))
            sm = np.sum(np.sqrt(si) * (tmp**2 + 1))
            result = (sm**2) / ((D - 1) ** 2)

            return result + self.f_global

        self.pure_objective_function = expanded_schaffer_f7

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        D = self.n_dim
        Z = self._batch_shift_rotate(X)

        S = xp.sqrt(Z[:, :-1] ** 2 + Z[:, 1:] ** 2)
        tmp = xp.sin(50 * (S**0.2))
        sm = xp.sum(xp.sqrt(S) * (tmp**2 + 1), axis=1)
        result = (sm**2) / ((D - 1) ** 2)

        return result + self.f_global


class ShiftedRotatedNonContRastrigin2022(CEC2022Function):
    """F4: Shifted and Rotated Non-Continuous Rastrigin's Function.

    Properties:
    - Multimodal (asymmetrical)
    - Non-separable
    - Huge number of local optima
    - Non-continuous
    """

    _spec = {
        "name": "Shifted and Rotated Non-Continuous Rastrigin's Function",
        "func_id": 4,
        "unimodal": False,
        "separable": False,
        "continuous": False,
    }

    def _create_objective_function(self) -> None:
        def non_cont_rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            shift = self._get_shift_vector()
            M = self._get_rotation_matrix()

            shifted = x - shift
            y = shifted.copy()
            mask = np.abs(shifted) > 0.5
            y[mask] = np.floor(2 * shifted[mask] + 0.5) / 2

            z = M @ y
            z = 0.0512 * z

            result = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)

            return result + self.f_global

        self.pure_objective_function = non_cont_rastrigin

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        D = self.n_dim

        shift = xp.asarray(self._get_shift_vector())
        M = xp.asarray(self._get_rotation_matrix())

        shifted = X - shift
        Y = xp.where(xp.abs(shifted) > 0.5, xp.floor(2 * shifted + 0.5) / 2, shifted)
        Z = Y @ M.T
        Z = 0.0512 * Z

        result = xp.sum(Z**2 - 10 * xp.cos(2 * math.pi * Z) + 10, axis=1)

        return result + self.f_global


class ShiftedRotatedLevy2022(CEC2022Function):
    """F5: Shifted and Rotated Levy Function.

    Properties:
    - Multimodal
    - Non-separable
    - Huge number of local optima
    """

    _spec = {
        "name": "Shifted and Rotated Levy Function",
        "func_id": 5,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def levy(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            w = 1.0 + 0.25 * (z - 1.0)
            term1 = np.sin(np.pi * w[0]) ** 2
            term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
            sm = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))

            return term1 + sm + term3 + self.f_global

        self.pure_objective_function = levy

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)

        W = 1.0 + 0.25 * (Z - 1.0)

        term1 = xp.sin(math.pi * W[:, 0]) ** 2
        term3 = (W[:, -1] - 1) ** 2 * (1 + xp.sin(2 * math.pi * W[:, -1]) ** 2)
        sm = xp.sum((W[:, :-1] - 1) ** 2 * (1 + 10 * xp.sin(math.pi * W[:, :-1] + 1) ** 2), axis=1)

        return term1 + sm + term3 + self.f_global


CEC2022_BASIC = [
    ShiftedRotatedZakharov2022,
    ShiftedRotatedRosenbrock2022,
    ShiftedRotatedExpandedSchafferF72022,
    ShiftedRotatedNonContRastrigin2022,
    ShiftedRotatedLevy2022,
]
