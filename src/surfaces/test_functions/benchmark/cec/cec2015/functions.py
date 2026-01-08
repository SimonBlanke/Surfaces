# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2015 benchmark functions F1-F15.

These functions are from the CEC 2015 Special Session on Learning-based
Real-Parameter Single Objective Optimization.

References
----------
Liang, J. J., Qu, B. Y., Suganthan, P. N., & Chen, Q. (2014).
Problem definitions and evaluation criteria for the CEC 2015
competition on learning-based real-parameter single objective optimization.
"""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2015 import CEC2015Function


# =============================================================================
# F1-F2: Unimodal Functions
# =============================================================================


class RotatedBentCigar2015(CEC2015Function):
    """F1: Rotated Bent Cigar Function.

    A unimodal function with a narrow ridge.
    """

    _spec = {
        "name": "Rotated Bent Cigar Function",
        "func_id": 1,
        "unimodal": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            return z[0]**2 + 1e6 * np.sum(z[1:]**2) + self.f_global
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        return Z[:, 0]**2 + 1e6 * xp.sum(Z[:, 1:]**2, axis=1) + self.f_global


class RotatedDiscus2015(CEC2015Function):
    """F2: Rotated Discus Function.

    A unimodal function with high condition number along one axis.
    """

    _spec = {
        "name": "Rotated Discus Function",
        "func_id": 2,
        "unimodal": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            return 1e6 * z[0]**2 + np.sum(z[1:]**2) + self.f_global
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        return 1e6 * Z[:, 0]**2 + xp.sum(Z[:, 1:]**2, axis=1) + self.f_global


# =============================================================================
# F3-F9: Multimodal Functions
# =============================================================================


class ShiftedRotatedWeierstrass2015(CEC2015Function):
    """F3: Shifted and Rotated Weierstrass Function.

    A multimodal function with many local optima.
    """

    _spec = {
        "name": "Shifted and Rotated Weierstrass Function",
        "func_id": 3,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x) * 0.5 / 100  # Scale to [-0.5, 0.5]

            a, b, k_max = 0.5, 3.0, 20
            D = self.n_dim
            result = 0.0
            for i in range(D):
                for k in range(k_max + 1):
                    result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))

            # Subtract the constant term
            const = D * sum(a**k * np.cos(np.pi * b**k) for k in range(k_max + 1))
            return result - const + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X) * 0.5 / 100

        a, b, k_max = 0.5, 3.0, 20
        D = self.n_dim
        n = X.shape[0]

        result = xp.zeros(n)
        for k in range(k_max + 1):
            result = result + a**k * xp.sum(xp.cos(2 * np.pi * b**k * (Z + 0.5)), axis=1)

        const = D * sum(a**kk * np.cos(np.pi * b**kk) for kk in range(k_max + 1))
        return result - const + self.f_global


class ShiftedRotatedSchwefel2015(CEC2015Function):
    """F4: Shifted and Rotated Schwefel's Function.

    A multimodal function with second-best optimum far from global optimum.
    """

    _spec = {
        "name": "Shifted and Rotated Schwefel's Function",
        "func_id": 4,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x) + 4.209687462275036e2

            D = self.n_dim
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

        D = self.n_dim
        n = X.shape[0]

        result = xp.zeros(n)
        for i in range(D):
            zi = Z[:, i]
            mask_normal = xp.abs(zi) <= 500
            mask_high = zi > 500
            mask_low = zi < -500

            # Normal case
            result = result + xp.where(mask_normal, zi * xp.sin(xp.sqrt(xp.abs(zi))), 0.0)

            # High case
            tmp_high = 500 - zi % 500
            result = result + xp.where(mask_high,
                tmp_high * xp.sin(xp.sqrt(xp.abs(tmp_high))) - (zi - 500)**2 / (10000 * D), 0.0)

            # Low case
            tmp_low = xp.abs(zi) % 500 - 500
            result = result + xp.where(mask_low,
                tmp_low * xp.sin(xp.sqrt(xp.abs(tmp_low))) - (zi + 500)**2 / (10000 * D), 0.0)

        return 418.9829 * D - result + self.f_global


class ShiftedRotatedKatsuura2015(CEC2015Function):
    """F5: Shifted and Rotated Katsuura Function.

    A multimodal function with many local optima.
    """

    _spec = {
        "name": "Shifted and Rotated Katsuura Function",
        "func_id": 5,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x) * 5 / 100  # Scale

            D = self.n_dim
            result = 1.0
            for i in range(D):
                inner = 0.0
                for j in range(1, 33):
                    inner += abs(2**j * z[i] - round(2**j * z[i])) / 2**j
                result *= (1 + (i + 1) * inner) ** (10.0 / D**1.2)

            return (10.0 / D**2) * (result - 1) + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X) * 5 / 100

        D = self.n_dim
        n = X.shape[0]

        result = xp.ones(n)
        for i in range(D):
            inner = xp.zeros(n)
            for j in range(1, 33):
                inner = inner + xp.abs(2**j * Z[:, i] - xp.round(2**j * Z[:, i])) / 2**j
            result = result * (1 + (i + 1) * inner) ** (10.0 / D**1.2)

        return (10.0 / D**2) * (result - 1) + self.f_global


class ShiftedRotatedHappyCat2015(CEC2015Function):
    """F6: Shifted and Rotated HappyCat Function.

    A multimodal function with narrow global optimum basin.
    """

    _spec = {
        "name": "Shifted and Rotated HappyCat Function",
        "func_id": 6,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x) * 5 / 100  # Scale

            D = self.n_dim
            sum_z = np.sum(z)
            sum_z2 = np.sum(z**2)

            return abs(sum_z2 - D)**0.25 + (0.5 * sum_z2 + sum_z) / D + 0.5 + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X) * 5 / 100

        D = self.n_dim
        sum_z = xp.sum(Z, axis=1)
        sum_z2 = xp.sum(Z**2, axis=1)

        return xp.abs(sum_z2 - D)**0.25 + (0.5 * sum_z2 + sum_z) / D + 0.5 + self.f_global


class ShiftedRotatedHGBat2015(CEC2015Function):
    """F7: Shifted and Rotated HGBat Function.

    A multimodal function derived from bat-inspired algorithm.
    """

    _spec = {
        "name": "Shifted and Rotated HGBat Function",
        "func_id": 7,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x) * 5 / 100  # Scale

            D = self.n_dim
            sum_z = np.sum(z)
            sum_z2 = np.sum(z**2)

            return abs(sum_z2**2 - sum_z**2)**0.5 + (0.5 * sum_z2 + sum_z) / D + 0.5 + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X) * 5 / 100

        D = self.n_dim
        sum_z = xp.sum(Z, axis=1)
        sum_z2 = xp.sum(Z**2, axis=1)

        return xp.abs(sum_z2**2 - sum_z**2)**0.5 + (0.5 * sum_z2 + sum_z) / D + 0.5 + self.f_global


class ExpandedGriewankRosenbrock2015(CEC2015Function):
    """F8: Expanded Griewank's plus Rosenbrock's Function.

    A composition of Griewank and Rosenbrock functions.
    """

    _spec = {
        "name": "Expanded Griewank's plus Rosenbrock's Function",
        "func_id": 8,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def griewank(z):
            return z**2 / 4000 - np.cos(z) + 1

        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x) * 5 / 100 + 1  # Shift to optimum at 1

            D = self.n_dim
            result = 0.0
            for i in range(D):
                tmp = 100 * (z[i]**2 - z[(i + 1) % D])**2 + (z[i] - 1)**2
                result += griewank(tmp)

            return result + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X) * 5 / 100 + 1

        D = self.n_dim
        n = X.shape[0]

        result = xp.zeros(n)
        for i in range(D):
            j = (i + 1) % D
            tmp = 100 * (Z[:, i]**2 - Z[:, j])**2 + (Z[:, i] - 1)**2
            result = result + tmp**2 / 4000 - xp.cos(tmp) + 1

        return result + self.f_global


class ExpandedScafferF62015(CEC2015Function):
    """F9: Expanded Scaffer's F6 Function.

    An expanded version of Scaffer's F6 function.
    """

    _spec = {
        "name": "Expanded Scaffer's F6 Function",
        "func_id": 9,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def scaffer_f6(x, y):
            tmp = x**2 + y**2
            return 0.5 + (np.sin(np.sqrt(tmp))**2 - 0.5) / (1 + 0.001 * tmp)**2

        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            D = self.n_dim
            result = 0.0
            for i in range(D):
                result += scaffer_f6(z[i], z[(i + 1) % D])

            return result + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)

        D = self.n_dim
        n = X.shape[0]

        result = xp.zeros(n)
        for i in range(D):
            j = (i + 1) % D
            tmp = Z[:, i]**2 + Z[:, j]**2
            result = result + 0.5 + (xp.sin(xp.sqrt(tmp))**2 - 0.5) / (1 + 0.001 * tmp)**2

        return result + self.f_global


# =============================================================================
# F10-F12: Hybrid Functions
# =============================================================================


class HybridFunction1_2015(CEC2015Function):
    """F10: Hybrid Function 1 (N=3).

    A hybrid of 3 basic functions with shuffled variables.
    """

    _spec = {
        "name": "Hybrid Function 1 (N=3)",
        "func_id": 10,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            # Shuffle variables
            shuffle = self._get_shuffle_indices()
            z = z[shuffle]

            D = self.n_dim
            p = [0.3, 0.3, 0.4]  # Proportions
            n1 = int(np.ceil(p[0] * D))
            n2 = int(np.ceil(p[1] * D))

            # Split into 3 groups
            z1, z2, z3 = z[:n1], z[n1:n1+n2], z[n1+n2:]

            # F1: Schwefel's (modified)
            f1 = np.sum(z1**2)

            # F2: Rastrigin
            f2 = np.sum(z2**2 - 10 * np.cos(2 * np.pi * z2) + 10)

            # F3: Elliptic
            D3 = len(z3)
            if D3 > 1:
                i = np.arange(D3)
                f3 = np.sum(10**(6 * i / (D3 - 1)) * z3**2)
            else:
                f3 = np.sum(z3**2)

            return f1 + f2 + f3 + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)

        shuffle = self._get_shuffle_indices()
        Z = Z[:, shuffle]

        D = self.n_dim
        p = [0.3, 0.3, 0.4]
        n1 = int(np.ceil(p[0] * D))
        n2 = int(np.ceil(p[1] * D))

        Z1, Z2, Z3 = Z[:, :n1], Z[:, n1:n1+n2], Z[:, n1+n2:]

        f1 = xp.sum(Z1**2, axis=1)
        f2 = xp.sum(Z2**2 - 10 * xp.cos(2 * np.pi * Z2) + 10, axis=1)

        D3 = Z3.shape[1]
        if D3 > 1:
            i = xp.arange(D3, dtype=X.dtype)
            f3 = xp.sum(10**(6 * i / (D3 - 1)) * Z3**2, axis=1)
        else:
            f3 = xp.sum(Z3**2, axis=1)

        return f1 + f2 + f3 + self.f_global


class HybridFunction2_2015(CEC2015Function):
    """F11: Hybrid Function 2 (N=4).

    A hybrid of 4 basic functions with shuffled variables.
    """

    _spec = {
        "name": "Hybrid Function 2 (N=4)",
        "func_id": 11,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            shuffle = self._get_shuffle_indices()
            z = z[shuffle]

            D = self.n_dim
            p = [0.2, 0.2, 0.3, 0.3]
            cuts = np.cumsum([int(np.ceil(pi * D)) for pi in p[:-1]])
            z1, z2, z3, z4 = np.split(z, cuts)

            # Griewank
            f1 = np.sum(z1**2) / 4000 - np.prod(np.cos(z1 / np.sqrt(np.arange(1, len(z1)+1)))) + 1

            # Weierstrass
            a, b, k_max = 0.5, 3.0, 20
            f2 = 0.0
            for i in range(len(z2)):
                for k in range(k_max + 1):
                    f2 += a**k * np.cos(2 * np.pi * b**k * (z2[i] + 0.5))
            f2 -= len(z2) * sum(a**k * np.cos(np.pi * b**k) for k in range(k_max + 1))

            # Rosenbrock
            f3 = np.sum(100 * (z3[1:] - z3[:-1]**2)**2 + (z3[:-1] - 1)**2) if len(z3) > 1 else 0

            # Rastrigin
            f4 = np.sum(z4**2 - 10 * np.cos(2 * np.pi * z4) + 10)

            return f1 + f2 + f3 + f4 + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)

        shuffle = self._get_shuffle_indices()
        Z = Z[:, shuffle]

        D = self.n_dim
        p = [0.2, 0.2, 0.3, 0.3]
        cuts = [int(np.ceil(p[0] * D)), int(np.ceil((p[0]+p[1]) * D)), int(np.ceil((p[0]+p[1]+p[2]) * D))]

        Z1, Z2, Z3, Z4 = Z[:, :cuts[0]], Z[:, cuts[0]:cuts[1]], Z[:, cuts[1]:cuts[2]], Z[:, cuts[2]:]
        n = X.shape[0]

        # Griewank
        D1 = Z1.shape[1]
        i1 = xp.arange(1, D1+1, dtype=X.dtype)
        f1 = xp.sum(Z1**2, axis=1) / 4000 - xp.prod(xp.cos(Z1 / xp.sqrt(i1)), axis=1) + 1

        # Weierstrass (simplified)
        a, b, k_max = 0.5, 3.0, 20
        D2 = Z2.shape[1]
        f2 = xp.zeros(n)
        for k in range(k_max + 1):
            f2 = f2 + a**k * xp.sum(xp.cos(2 * np.pi * b**k * (Z2 + 0.5)), axis=1)
        f2 = f2 - D2 * sum(a**kk * np.cos(np.pi * b**kk) for kk in range(k_max + 1))

        # Rosenbrock
        D3 = Z3.shape[1]
        if D3 > 1:
            f3 = xp.sum(100 * (Z3[:, 1:] - Z3[:, :-1]**2)**2 + (Z3[:, :-1] - 1)**2, axis=1)
        else:
            f3 = xp.zeros(n)

        # Rastrigin
        f4 = xp.sum(Z4**2 - 10 * xp.cos(2 * np.pi * Z4) + 10, axis=1)

        return f1 + f2 + f3 + f4 + self.f_global


class HybridFunction3_2015(CEC2015Function):
    """F12: Hybrid Function 3 (N=5).

    A hybrid of 5 basic functions with shuffled variables.
    """

    _spec = {
        "name": "Hybrid Function 3 (N=5)",
        "func_id": 12,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            shuffle = self._get_shuffle_indices()
            z = z[shuffle]

            D = self.n_dim
            p = [0.1, 0.2, 0.2, 0.2, 0.3]
            cuts = np.cumsum([int(np.ceil(pi * D)) for pi in p[:-1]])
            z1, z2, z3, z4, z5 = np.split(z, cuts)

            # Bent Cigar
            f1 = z1[0]**2 + 1e6 * np.sum(z1[1:]**2) if len(z1) > 1 else np.sum(z1**2)

            # HGBat
            sum_z2 = np.sum(z2)
            sum_z2_sq = np.sum(z2**2)
            D2 = len(z2)
            f2 = abs(sum_z2_sq**2 - sum_z2**2)**0.5 + (0.5*sum_z2_sq + sum_z2)/D2 + 0.5 if D2 > 0 else 0

            # Rastrigin
            f3 = np.sum(z3**2 - 10 * np.cos(2 * np.pi * z3) + 10)

            # Rosenbrock
            f4 = np.sum(100 * (z4[1:] - z4[:-1]**2)**2 + (z4[:-1] - 1)**2) if len(z4) > 1 else 0

            # Elliptic
            D5 = len(z5)
            if D5 > 1:
                i = np.arange(D5)
                f5 = np.sum(10**(6 * i / (D5 - 1)) * z5**2)
            else:
                f5 = np.sum(z5**2)

            return f1 + f2 + f3 + f4 + f5 + self.f_global

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)

        shuffle = self._get_shuffle_indices()
        Z = Z[:, shuffle]

        D = self.n_dim
        n = X.shape[0]
        p = [0.1, 0.2, 0.2, 0.2, 0.3]
        c = [int(np.ceil(p[0]*D)), int(np.ceil((p[0]+p[1])*D)),
             int(np.ceil((p[0]+p[1]+p[2])*D)), int(np.ceil((p[0]+p[1]+p[2]+p[3])*D))]

        Z1, Z2, Z3, Z4, Z5 = Z[:, :c[0]], Z[:, c[0]:c[1]], Z[:, c[1]:c[2]], Z[:, c[2]:c[3]], Z[:, c[3]:]

        # Bent Cigar
        D1 = Z1.shape[1]
        f1 = Z1[:, 0]**2 + 1e6 * xp.sum(Z1[:, 1:]**2, axis=1) if D1 > 1 else xp.sum(Z1**2, axis=1)

        # HGBat
        D2 = Z2.shape[1]
        sum_z2 = xp.sum(Z2, axis=1)
        sum_z2_sq = xp.sum(Z2**2, axis=1)
        f2 = xp.abs(sum_z2_sq**2 - sum_z2**2)**0.5 + (0.5*sum_z2_sq + sum_z2)/D2 + 0.5 if D2 > 0 else xp.zeros(n)

        # Rastrigin
        f3 = xp.sum(Z3**2 - 10 * xp.cos(2 * np.pi * Z3) + 10, axis=1)

        # Rosenbrock
        D4 = Z4.shape[1]
        f4 = xp.sum(100 * (Z4[:, 1:] - Z4[:, :-1]**2)**2 + (Z4[:, :-1] - 1)**2, axis=1) if D4 > 1 else xp.zeros(n)

        # Elliptic
        D5 = Z5.shape[1]
        if D5 > 1:
            i5 = xp.arange(D5, dtype=X.dtype)
            f5 = xp.sum(10**(6 * i5 / (D5 - 1)) * Z5**2, axis=1)
        else:
            f5 = xp.sum(Z5**2, axis=1)

        return f1 + f2 + f3 + f4 + f5 + self.f_global


# =============================================================================
# F13-F15: Composition Functions
# =============================================================================


class CompositionFunction1_2015(CEC2015Function):
    """F13: Composition Function 1 (N=5).

    A composition of 5 basic functions.
    """

    _spec = {
        "name": "Composition Function 1 (N=5)",
        "func_id": 13,
        "unimodal": False,
        "separable": False,
    }

    _n_components = 5
    _sigmas = [10, 20, 30, 40, 50]
    _lambdas = [1, 1, 1, 1, 1]

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._composition_eval(x) + self.f_global
        self.pure_objective_function = objective

    def _composition_eval(self, x):
        """Evaluate composition function."""
        data = self._load_data()
        optima = data.get(f"comp_optima_{self.func_id}", np.zeros((self._n_components, self.n_dim)))

        D = self.n_dim
        weights = np.zeros(self._n_components)

        for i in range(self._n_components):
            diff = x - optima[i]
            weights[i] = np.exp(-np.sum(diff**2) / (2 * D * self._sigmas[i]**2))

        # Normalize weights
        weights /= np.sum(weights) + 1e-10

        result = 0.0
        for i in range(self._n_components):
            # Get rotation matrix for component
            R = data.get(f"rotation_{self.func_id}_{i+1}", np.eye(D))
            z = R @ (x - optima[i])

            # Component functions
            if i == 0:  # Rosenbrock
                f = np.sum(100 * (z[1:] - z[:-1]**2)**2 + (z[:-1] - 1)**2) if D > 1 else 0
            elif i == 1:  # Elliptic
                idx = np.arange(D)
                f = np.sum(10**(6 * idx / (D - 1)) * z**2) if D > 1 else np.sum(z**2)
            elif i == 2:  # Bent Cigar
                f = z[0]**2 + 1e6 * np.sum(z[1:]**2) if D > 1 else np.sum(z**2)
            elif i == 3:  # Discus
                f = 1e6 * z[0]**2 + np.sum(z[1:]**2) if D > 1 else np.sum(z**2)
            else:  # Elliptic again
                idx = np.arange(D)
                f = np.sum(10**(6 * idx / (D - 1)) * z**2) if D > 1 else np.sum(z**2)

            result += weights[i] * self._lambdas[i] * f

        return result

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        results = xp.zeros(n)
        for i in range(n):
            results[i] = self._composition_eval(np.array(X[i]))
        return results + self.f_global


class CompositionFunction2_2015(CEC2015Function):
    """F14: Composition Function 2 (N=3).

    A composition of 3 basic functions.
    """

    _spec = {
        "name": "Composition Function 2 (N=3)",
        "func_id": 14,
        "unimodal": False,
        "separable": False,
    }

    _n_components = 3
    _sigmas = [20, 20, 20]
    _lambdas = [1, 10, 1]

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._composition_eval(x) + self.f_global
        self.pure_objective_function = objective

    def _composition_eval(self, x):
        """Evaluate composition function."""
        data = self._load_data()
        optima = data.get(f"comp_optima_{self.func_id}", np.zeros((self._n_components, self.n_dim)))

        D = self.n_dim
        weights = np.zeros(self._n_components)

        for i in range(self._n_components):
            diff = x - optima[i]
            weights[i] = np.exp(-np.sum(diff**2) / (2 * D * self._sigmas[i]**2))

        weights /= np.sum(weights) + 1e-10

        result = 0.0
        for i in range(self._n_components):
            R = data.get(f"rotation_{self.func_id}_{i+1}", np.eye(D))
            z = R @ (x - optima[i])

            # Component functions: Schwefel, Rastrigin, HGBat
            if i == 0:  # Schwefel 1.2
                f = np.sum([np.sum(z[:j+1])**2 for j in range(D)])
            elif i == 1:  # Rastrigin
                f = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
            else:  # HGBat
                sum_z = np.sum(z)
                sum_z2 = np.sum(z**2)
                f = abs(sum_z2**2 - sum_z**2)**0.5 + (0.5 * sum_z2 + sum_z) / D + 0.5

            result += weights[i] * self._lambdas[i] * f

        return result

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        results = xp.zeros(n)
        for i in range(n):
            results[i] = self._composition_eval(np.array(X[i]))
        return results + self.f_global


class CompositionFunction3_2015(CEC2015Function):
    """F15: Composition Function 3 (N=5).

    A composition of 5 basic functions.
    """

    _spec = {
        "name": "Composition Function 3 (N=5)",
        "func_id": 15,
        "unimodal": False,
        "separable": False,
    }

    _n_components = 5
    _sigmas = [10, 20, 30, 40, 50]
    _lambdas = [10, 10, 2.5, 25, 1e-6]

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            return self._composition_eval(x) + self.f_global
        self.pure_objective_function = objective

    def _composition_eval(self, x):
        """Evaluate composition function."""
        data = self._load_data()
        optima = data.get(f"comp_optima_{self.func_id}", np.zeros((self._n_components, self.n_dim)))

        D = self.n_dim
        weights = np.zeros(self._n_components)

        for i in range(self._n_components):
            diff = x - optima[i]
            weights[i] = np.exp(-np.sum(diff**2) / (2 * D * self._sigmas[i]**2))

        weights /= np.sum(weights) + 1e-10

        result = 0.0
        for i in range(self._n_components):
            R = data.get(f"rotation_{self.func_id}_{i+1}", np.eye(D))
            z = R @ (x - optima[i])

            # Diverse component functions
            if i == 0:  # HGBat
                sum_z = np.sum(z)
                sum_z2 = np.sum(z**2)
                f = abs(sum_z2**2 - sum_z**2)**0.5 + (0.5 * sum_z2 + sum_z) / D + 0.5
            elif i == 1:  # Rastrigin
                f = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
            elif i == 2:  # Schwefel
                f = np.sum([np.sum(z[:j+1])**2 for j in range(D)])
            elif i == 3:  # Weierstrass
                a, b, k_max = 0.5, 3.0, 20
                f = 0.0
                for j in range(D):
                    for k in range(k_max + 1):
                        f += a**k * np.cos(2 * np.pi * b**k * (z[j] + 0.5))
                f -= D * sum(a**k * np.cos(np.pi * b**k) for k in range(k_max + 1))
            else:  # Elliptic
                idx = np.arange(D)
                f = np.sum(10**(6 * idx / (D - 1)) * z**2) if D > 1 else np.sum(z**2)

            result += weights[i] * self._lambdas[i] * f

        return result

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n = X.shape[0]
        results = xp.zeros(n)
        for i in range(n):
            results[i] = self._composition_eval(np.array(X[i]))
        return results + self.f_global


# =============================================================================
# All CEC 2015 functions
# =============================================================================

CEC2015_ALL = [
    RotatedBentCigar2015,
    RotatedDiscus2015,
    ShiftedRotatedWeierstrass2015,
    ShiftedRotatedSchwefel2015,
    ShiftedRotatedKatsuura2015,
    ShiftedRotatedHappyCat2015,
    ShiftedRotatedHGBat2015,
    ExpandedGriewankRosenbrock2015,
    ExpandedScafferF62015,
    HybridFunction1_2015,
    HybridFunction2_2015,
    HybridFunction3_2015,
    CompositionFunction1_2015,
    CompositionFunction2_2015,
    CompositionFunction3_2015,
]

CEC2015_UNIMODAL = [
    RotatedBentCigar2015,
    RotatedDiscus2015,
]

CEC2015_MULTIMODAL = [
    ShiftedRotatedWeierstrass2015,
    ShiftedRotatedSchwefel2015,
    ShiftedRotatedKatsuura2015,
    ShiftedRotatedHappyCat2015,
    ShiftedRotatedHGBat2015,
    ExpandedGriewankRosenbrock2015,
    ExpandedScafferF62015,
]

CEC2015_HYBRID = [
    HybridFunction1_2015,
    HybridFunction2_2015,
    HybridFunction3_2015,
]

CEC2015_COMPOSITION = [
    CompositionFunction1_2015,
    CompositionFunction2_2015,
    CompositionFunction3_2015,
]
