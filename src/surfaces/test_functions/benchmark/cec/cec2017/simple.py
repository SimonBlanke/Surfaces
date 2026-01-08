# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2017 Simple Benchmark Functions (F1-F10)."""

import math
from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2017 import CEC2017Function


class ShiftedRotatedBentCigar(CEC2017Function):
    """F1: Shifted and Rotated Bent Cigar Function.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Bent Cigar Function",
        "func_id": 1,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    # Function sheet attributes
    latex_formula = r"f(\vec{z}) = z_1^2 + 10^6 \sum_{i=2}^{n} z_i^2 \quad \text{where } \vec{z} = M(\vec{x} - \vec{o})"
    tagline = (
        "A shifted and rotated ill-conditioned function. "
        "One dimension dominates, creating a narrow valley in transformed space."
    )
    display_bounds = (-100.0, 100.0)
    display_projection = {"fixed_value": 0.0}
    reference = "CEC 2017 Competition"
    reference_url = "https://github.com/P-N-Suganthan/CEC2017-BoundConstrained"

    def _create_objective_function(self) -> None:
        def bent_cigar(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2) + self.f_global

        self.pure_objective_function = bent_cigar

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        return Z[:, 0] ** 2 + 1e6 * xp.sum(Z[:, 1:] ** 2, axis=1) + self.f_global


class ShiftedRotatedSumDiffPow(CEC2017Function):
    """F2: Shifted and Rotated Sum of Different Power Function (DEPRECATED).

    Note: This function has been deprecated from the CEC 2017 benchmark suite.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Sum of Different Power Function",
        "func_id": 2,
        "unimodal": True,
        "separable": False,
        "deprecated": True,
    }

    def _create_objective_function(self) -> None:
        def sum_diff_pow(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            D = self.n_dim
            result = sum(abs(z[i]) ** (i + 1) for i in range(D))
            return result + self.f_global

        self.pure_objective_function = sum_diff_pow

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        D = self.n_dim
        Z = self._batch_shift_rotate(X)

        # exponents: i+1 for i = 0 to D-1
        exponents = xp.arange(1, D + 1, dtype=X.dtype)
        result = xp.sum(xp.abs(Z) ** exponents, axis=1)
        return result + self.f_global


class ShiftedRotatedZakharov(CEC2017Function):
    """F3: Shifted and Rotated Zakharov Function.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Zakharov Function",
        "func_id": 3,
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
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        D = self.n_dim
        Z = self._batch_shift_rotate(X)

        # i = 1 to D
        i = xp.arange(1, D + 1, dtype=X.dtype)

        sum1 = xp.sum(Z**2, axis=1)
        sum2 = xp.sum(0.5 * i * Z, axis=1)
        return sum1 + sum2**2 + sum2**4 + self.f_global


class ShiftedRotatedRosenbrock(CEC2017Function):
    """F4: Shifted and Rotated Rosenbrock's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Rosenbrock's Function",
        "func_id": 4,
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
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        Z = 0.02048 * Z + 1.0

        # Rosenbrock: sum of 100*(z[i]^2 - z[i+1])^2 + (z[i] - 1)^2
        result = xp.sum(100 * (Z[:, :-1] ** 2 - Z[:, 1:]) ** 2 + (Z[:, :-1] - 1) ** 2, axis=1)
        return result + self.f_global


class ShiftedRotatedRastrigin(CEC2017Function):
    """F5: Shifted and Rotated Rastrigin's Function.

    Properties:
    - Highly multimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Rastrigin's Function",
        "func_id": 5,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = 0.0512 * z

            D = self.n_dim
            result = 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

            return result + self.f_global

        self.pure_objective_function = rastrigin

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        D = self.n_dim
        Z = self._batch_shift_rotate(X)
        Z = 0.0512 * Z

        result = 10 * D + xp.sum(Z**2 - 10 * xp.cos(2 * math.pi * Z), axis=1)
        return result + self.f_global


class ShiftedRotatedSchafferF7(CEC2017Function):
    """F6: Shifted and Rotated Schaffer's F7 Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Schaffer's F7 Function",
        "func_id": 6,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def schaffers_f7(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            D = self.n_dim
            si = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
            tmp = np.sin(50 * (si**0.2))
            sm = np.sum(np.sqrt(si) * (tmp**2 + 1))
            result = (sm**2) / ((D - 1) ** 2)

            return result + self.f_global

        self.pure_objective_function = schaffers_f7

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        D = self.n_dim
        Z = self._batch_shift_rotate(X)

        # s[i] = sqrt(z[i]^2 + z[i+1]^2)
        S = xp.sqrt(Z[:, :-1] ** 2 + Z[:, 1:] ** 2)
        tmp = xp.sin(50 * (S**0.2))
        sm = xp.sum(xp.sqrt(S) * (tmp**2 + 1), axis=1)
        result = (sm**2) / ((D - 1) ** 2)

        return result + self.f_global


class ShiftedRotatedLunacekBiRastrigin(CEC2017Function):
    """F7: Shifted and Rotated Lunacek Bi-Rastrigin's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Two global optima
    """

    _spec = {
        "name": "Shifted and Rotated Lunacek Bi-Rastrigin's Function",
        "func_id": 7,
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
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        D = self.n_dim

        shift = xp.asarray(self._get_shift_vector())
        M = xp.asarray(self._get_rotation_matrix())

        mu0 = 2.5
        s = 1 - 1 / (2 * math.sqrt(D + 20) - 8.2)
        mu1 = -math.sqrt((mu0**2 - 1) / s)

        Y = 0.1 * (X - shift)
        TMPX = 2 * Y

        # tmpx[shift < 0] *= -1
        sign_mask = xp.where(shift < 0, -1.0, 1.0)
        TMPX = TMPX * sign_mask

        Z = TMPX.copy()
        TMPX = TMPX + mu0

        t1 = xp.sum((TMPX - mu0) ** 2, axis=1)
        t2 = s * xp.sum((TMPX - mu1) ** 2, axis=1) + D

        # y = M @ z for each point
        Y_rot = Z @ M.T
        t = xp.sum(xp.cos(2 * math.pi * Y_rot), axis=1)

        result = xp.minimum(t1, t2) + 10 * (D - t)

        return result + self.f_global


class ShiftedRotatedNonContRastrigin(CEC2017Function):
    """F8: Shifted and Rotated Non-Continuous Rastrigin's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Non-continuous
    """

    _spec = {
        "name": "Shifted and Rotated Non-Continuous Rastrigin's Function",
        "func_id": 8,
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
            x_mod = x.copy()
            mask = np.abs(shifted) > 0.5
            x_mod[mask] = (shift + np.floor(2 * shifted + 0.5) * 0.5)[mask]

            z = 0.0512 * shifted
            z = M @ z

            result = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)

            return result + self.f_global

        self.pure_objective_function = non_cont_rastrigin

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)

        shift = xp.asarray(self._get_shift_vector())
        M = xp.asarray(self._get_rotation_matrix())

        shifted = X - shift
        Z = 0.0512 * shifted
        Z = Z @ M.T

        result = xp.sum(Z**2 - 10 * xp.cos(2 * math.pi * Z) + 10, axis=1)

        return result + self.f_global


class ShiftedRotatedLevy(CEC2017Function):
    """F9: Shifted and Rotated Levy Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Levy Function",
        "func_id": 9,
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
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)

        W = 1.0 + 0.25 * (Z - 1.0)

        term1 = xp.sin(math.pi * W[:, 0]) ** 2
        term3 = (W[:, -1] - 1) ** 2 * (1 + xp.sin(2 * math.pi * W[:, -1]) ** 2)
        sm = xp.sum(
            (W[:, :-1] - 1) ** 2 * (1 + 10 * xp.sin(math.pi * W[:, :-1] + 1) ** 2),
            axis=1,
        )

        return term1 + sm + term3 + self.f_global


class ShiftedRotatedSchwefel(CEC2017Function):
    """F10: Shifted and Rotated Schwefel's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Deceptive
    """

    _spec = {
        "name": "Shifted and Rotated Schwefel's Function",
        "func_id": 10,
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
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        D = self.n_dim

        Z = self._batch_shift_rotate(X)
        Z = 10.0 * Z + 420.9687462275036

        # Case 1: abs(z) <= 500
        case1 = xp.abs(Z) <= 500
        term1 = Z * xp.sin(xp.sqrt(xp.abs(Z)))

        # Case 2: z > 500
        case2 = Z > 500
        zm2 = 500 - Z % 500
        term2 = zm2 * xp.sin(xp.sqrt(xp.abs(zm2))) - (Z - 500) ** 2 / (10000 * D)

        # Case 3: z < -500 (else branch)
        zm3 = xp.abs(Z) % 500 - 500
        term3 = zm3 * xp.sin(xp.sqrt(xp.abs(zm3))) - (Z + 500) ** 2 / (10000 * D)

        contrib = xp.where(case1, term1, xp.where(case2, term2, term3))
        result = 418.9829 * D - xp.sum(contrib, axis=1)

        return result + self.f_global
