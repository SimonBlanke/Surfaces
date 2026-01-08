# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2005 Multimodal Functions (F6-F14)."""

import math
from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2005 import CEC2005Function, CEC2005NonRotatedFunction


# =============================================================================
# F6: Shifted Rosenbrock's Function
# =============================================================================


class ShiftedRosenbrock(CEC2005NonRotatedFunction):
    """F6: Shifted Rosenbrock's Function.

    f(x) = sum_i(100*(z_i^2 - z_{i+1})^2 + (z_i - 1)^2) + f_bias
    where z = x - o + 1

    Properties:
    - Multimodal (narrow valley)
    - Non-separable
    - Scalable
    - Bounds: [-100, 100]^D
    """

    _spec = {
        "name": "Shifted Rosenbrock's Function",
        "func_id": 6,
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
        "convex": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def rosenbrock(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id) + 1  # Shift to have optimum at z=1

            result = 0.0
            for i in range(self.n_dim - 1):
                result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2

            return float(result + self.f_global)

        self.pure_objective_function = rosenbrock

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id) + 1

        # z_i^2 - z_{i+1} and (z_i - 1)^2
        Z_sq = Z[:, :-1] ** 2
        Z_next = Z[:, 1:]
        Z_minus1 = Z[:, :-1] - 1

        result = xp.sum(100 * (Z_sq - Z_next) ** 2 + Z_minus1**2, axis=1)
        return result + self.f_global


# =============================================================================
# F7: Shifted Rotated Griewank's Function without Bounds
# =============================================================================


class ShiftedRotatedGriewank(CEC2005Function):
    """F7: Shifted Rotated Griewank's Function without Bounds.

    f(x) = sum(z_i^2/4000) - prod(cos(z_i/sqrt(i+1))) + 1 + f_bias
    where z = M @ (x - o)

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    - Bounds: [-100, 100]^D (actually unbounded in spec)
    """

    _spec = {
        "name": "Shifted Rotated Griewank's Function",
        "func_id": 7,
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
        "convex": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def griewank(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            # Scale by 600/100 (as per CEC2005 spec)
            z = z * 600 / 100

            sum_term = np.sum(z**2) / 4000
            prod_term = np.prod(
                [np.cos(z[i] / np.sqrt(i + 1)) for i in range(self.n_dim)]
            )

            return float(sum_term - prod_term + 1 + self.f_global)

        self.pure_objective_function = griewank

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        Z = Z * 600 / 100

        sum_term = xp.sum(Z**2, axis=1) / 4000

        # sqrt(i+1) for i = 0..D-1
        sqrt_i = xp.sqrt(xp.arange(1, self.n_dim + 1, dtype=X.dtype))
        prod_term = xp.prod(xp.cos(Z / sqrt_i), axis=1)

        return sum_term - prod_term + 1 + self.f_global


# =============================================================================
# F8: Shifted Rotated Ackley's Function with Global Optimum on Bounds
# =============================================================================


class ShiftedRotatedAckley(CEC2005Function):
    """F8: Shifted Rotated Ackley's Function with Global Optimum on Bounds.

    f(x) = -20*exp(-0.2*sqrt(sum(z_i^2)/D)) - exp(sum(cos(2*pi*z_i))/D) + 20 + e + f_bias
    where z = M @ (x - o)

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    - Bounds: [-32, 32]^D
    - Global optimum on bounds
    """

    _spec = {
        "name": "Shifted Rotated Ackley's Function",
        "func_id": 8,
        "default_bounds": (-32.0, 32.0),
        "unimodal": False,
        "convex": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def ackley(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            D = self.n_dim
            sum_sq = np.sum(z**2) / D
            sum_cos = np.sum(np.cos(2 * np.pi * z)) / D

            result = -20 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20 + np.e

            return float(result + self.f_global)

        self.pure_objective_function = ackley

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)

        D = self.n_dim
        sum_sq = xp.sum(Z**2, axis=1) / D
        sum_cos = xp.sum(xp.cos(2 * np.pi * Z), axis=1) / D

        result = -20 * xp.exp(-0.2 * xp.sqrt(sum_sq)) - xp.exp(sum_cos) + 20 + np.e

        return result + self.f_global


# =============================================================================
# F9: Shifted Rastrigin's Function
# =============================================================================


class ShiftedRastrigin(CEC2005NonRotatedFunction):
    """F9: Shifted Rastrigin's Function.

    f(x) = sum(z_i^2 - 10*cos(2*pi*z_i) + 10) + f_bias
    where z = x - o

    Properties:
    - Multimodal (highly)
    - Separable
    - Scalable
    - Bounds: [-5, 5]^D
    """

    _spec = {
        "name": "Shifted Rastrigin's Function",
        "func_id": 9,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "convex": False,
        "separable": True,
    }

    def _create_objective_function(self) -> None:
        def rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)

            result = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)

            return float(result + self.f_global)

        self.pure_objective_function = rastrigin

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)

        result = xp.sum(Z**2 - 10 * xp.cos(2 * np.pi * Z) + 10, axis=1)
        return result + self.f_global


# =============================================================================
# F10: Shifted Rotated Rastrigin's Function
# =============================================================================


class ShiftedRotatedRastrigin(CEC2005Function):
    """F10: Shifted Rotated Rastrigin's Function.

    f(x) = sum(z_i^2 - 10*cos(2*pi*z_i) + 10) + f_bias
    where z = M @ (x - o)

    Properties:
    - Multimodal (highly)
    - Non-separable
    - Scalable
    - Bounds: [-5, 5]^D
    """

    _spec = {
        "name": "Shifted Rotated Rastrigin's Function",
        "func_id": 10,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "convex": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            result = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)

            return float(result + self.f_global)

        self.pure_objective_function = rastrigin

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)

        result = xp.sum(Z**2 - 10 * xp.cos(2 * np.pi * Z) + 10, axis=1)
        return result + self.f_global


# =============================================================================
# F11: Shifted Rotated Weierstrass Function
# =============================================================================


class ShiftedRotatedWeierstrass(CEC2005Function):
    """F11: Shifted Rotated Weierstrass Function.

    f(x) = sum_i(sum_k(a^k * cos(2*pi*b^k*(z_i+0.5)))) - D*sum_k(a^k*cos(pi*b^k)) + f_bias
    where z = M @ (x - o), a = 0.5, b = 3, k = 0..k_max

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    - Bounds: [-0.5, 0.5]^D
    """

    _spec = {
        "name": "Shifted Rotated Weierstrass Function",
        "func_id": 11,
        "default_bounds": (-0.5, 0.5),
        "unimodal": False,
        "convex": False,
        "separable": False,
    }

    # Weierstrass parameters
    _a = 0.5
    _b = 3
    _k_max = 20

    def _create_objective_function(self) -> None:
        a, b, k_max = self._a, self._b, self._k_max

        # Precompute constant term
        const_term = 0.0
        for k in range(k_max + 1):
            const_term += (a**k) * np.cos(np.pi * (b**k))

        def weierstrass(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            # Scale from [-100, 100] to [-0.5, 0.5]
            z = z * 0.5 / 100

            result = 0.0
            for i in range(self.n_dim):
                for k in range(k_max + 1):
                    result += (a**k) * np.cos(2 * np.pi * (b**k) * (z[i] + 0.5))

            result -= self.n_dim * const_term

            return float(result + self.f_global)

        self.pure_objective_function = weierstrass

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        a, b, k_max = self._a, self._b, self._k_max

        Z = self._batch_shift_rotate(X)
        Z = Z * 0.5 / 100

        # k values: 0, 1, ..., k_max
        k = xp.arange(k_max + 1, dtype=X.dtype)
        a_k = xp.power(a, k)  # (k_max+1,)
        b_k = xp.power(float(b), k)  # (k_max+1,)

        # Z: (n_points, D)
        # We need sum over D and k: sum_i sum_k a^k * cos(2*pi*b^k*(z_i+0.5))

        # Expand for broadcasting: Z[:, :, None] * b_k[None, None, :]
        Z_expanded = Z[:, :, None] + 0.5  # (n_points, D, 1)
        terms = a_k * xp.cos(2 * np.pi * b_k * Z_expanded)  # (n_points, D, k_max+1)
        result = xp.sum(xp.sum(terms, axis=2), axis=1)

        # Constant term
        const_term = xp.sum(a_k * xp.cos(np.pi * b_k))
        result = result - self.n_dim * const_term

        return result + self.f_global


# =============================================================================
# F12: Schwefel's Problem 2.13
# =============================================================================


class SchwefelProblem213(CEC2005NonRotatedFunction):
    """F12: Schwefel's Problem 2.13.

    f(x) = sum_i(A_i - B_i(x))^2 + f_bias
    where A_i = sum_j(a_ij*sin(alpha_j) + b_ij*cos(alpha_j))
          B_i(x) = sum_j(a_ij*sin(x_j) + b_ij*cos(x_j))

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    - Bounds: [-pi, pi]^D (or [-100, 100] in some versions)
    """

    _spec = {
        "name": "Schwefel's Problem 2.13",
        "func_id": 12,
        "default_bounds": (-np.pi, np.pi),
        "unimodal": False,
        "convex": False,
        "separable": False,
    }

    def _get_alpha(self) -> np.ndarray:
        """Get the alpha vector (optimal x)."""
        data = self._load_data()
        return data.get("shift_12", np.zeros(self.n_dim))

    def _get_a_matrix(self) -> np.ndarray:
        """Get the A matrix."""
        data = self._load_data()
        return data.get("a_matrix_12", np.zeros((self.n_dim, self.n_dim)))

    def _get_b_matrix(self) -> np.ndarray:
        """Get the B matrix."""
        data = self._load_data()
        return data.get("b_matrix_12", np.zeros((self.n_dim, self.n_dim)))

    def _create_objective_function(self) -> None:
        alpha = self._get_alpha()
        A = self._get_a_matrix()
        B = self._get_b_matrix()

        # Precompute A_i = sum_j(a_ij*sin(alpha_j) + b_ij*cos(alpha_j))
        A_target = A @ np.sin(alpha) + B @ np.cos(alpha)

        def schwefel_213(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)

            # B_i(x) = sum_j(a_ij*sin(x_j) + b_ij*cos(x_j))
            B_x = A @ np.sin(x) + B @ np.cos(x)

            result = np.sum((A_target - B_x) ** 2)

            return float(result + self.f_global)

        self.pure_objective_function = schwefel_213

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        alpha = xp.asarray(self._get_alpha())
        A_mat = xp.asarray(self._get_a_matrix())
        B_mat = xp.asarray(self._get_b_matrix())

        # A_target: (D,)
        A_target = A_mat @ xp.sin(alpha) + B_mat @ xp.cos(alpha)

        # X: (n_points, D)
        # sin(X): (n_points, D), cos(X): (n_points, D)
        # A_mat @ sin(X).T: (D, n_points) -> transpose -> (n_points, D)
        sin_X = xp.sin(X)
        cos_X = xp.cos(X)
        B_x = (A_mat @ sin_X.T + B_mat @ cos_X.T).T  # (n_points, D)

        result = xp.sum((A_target - B_x) ** 2, axis=1)

        return result + self.f_global


# =============================================================================
# F13: Expanded Extended Griewank's plus Rosenbrock's Function (EF8F2)
# =============================================================================


class ExpandedGriewankRosenbrock(CEC2005Function):
    """F13: Expanded Extended Griewank's plus Rosenbrock's Function (EF8F2).

    This is a composition of Rosenbrock and Griewank functions.
    f(x) = sum_i(F8(F2(z_i, z_{i+1}))) + f_bias
    where F2 is Rosenbrock component and F8 is Griewank component

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    - Bounds: [-3, 1]^D (or [-5, 5])
    """

    _spec = {
        "name": "Expanded Extended Griewank's plus Rosenbrock's Function",
        "func_id": 13,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "convex": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def ef8f2(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 5 / 100 + 1  # Scale and shift

            result = 0.0
            for i in range(self.n_dim):
                # Rosenbrock term
                z_i = z[i]
                z_next = z[(i + 1) % self.n_dim]
                f2 = 100 * (z_i**2 - z_next) ** 2 + (z_i - 1) ** 2

                # Griewank of Rosenbrock
                result += f2**2 / 4000 - np.cos(f2) + 1

            return float(result + self.f_global)

        self.pure_objective_function = ef8f2

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        Z = Z * 5 / 100 + 1

        # Wrap-around: z_next = roll(Z, -1)
        Z_next = xp.roll(Z, -1, axis=1)

        # Rosenbrock: 100*(z^2 - z_next)^2 + (z - 1)^2
        F2 = 100 * (Z**2 - Z_next) ** 2 + (Z - 1) ** 2

        # Griewank of Rosenbrock: f^2/4000 - cos(f) + 1
        result = xp.sum(F2**2 / 4000 - xp.cos(F2) + 1, axis=1)

        return result + self.f_global


# =============================================================================
# F14: Shifted Rotated Expanded Scaffer's F6 Function
# =============================================================================


class ShiftedRotatedExpandedScaffer(CEC2005Function):
    """F14: Shifted Rotated Expanded Scaffer's F6 Function.

    f(x) = sum_i(g(z_i, z_{i+1})) + f_bias
    where g(x,y) = 0.5 + (sin^2(sqrt(x^2+y^2)) - 0.5) / (1 + 0.001*(x^2+y^2))^2

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    - Bounds: [-100, 100]^D
    """

    _spec = {
        "name": "Shifted Rotated Expanded Scaffer's F6 Function",
        "func_id": 14,
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
        "convex": False,
        "separable": False,
    }

    def _scaffer_f6(self, x: float, y: float) -> float:
        """Scaffer's F6 function for two variables."""
        sum_sq = x**2 + y**2
        sin_term = np.sin(np.sqrt(sum_sq)) ** 2 - 0.5
        denom = (1 + 0.001 * sum_sq) ** 2
        return 0.5 + sin_term / denom

    def _create_objective_function(self) -> None:
        def expanded_scaffer(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            result = 0.0
            for i in range(self.n_dim):
                z_i = z[i]
                z_next = z[(i + 1) % self.n_dim]
                result += self._scaffer_f6(z_i, z_next)

            return float(result + self.f_global)

        self.pure_objective_function = expanded_scaffer

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)

        # Wrap-around
        Z_next = xp.roll(Z, -1, axis=1)

        # Scaffer F6: 0.5 + (sin^2(sqrt(x^2+y^2)) - 0.5) / (1 + 0.001*(x^2+y^2))^2
        sum_sq = Z**2 + Z_next**2
        sin_term = xp.sin(xp.sqrt(sum_sq)) ** 2 - 0.5
        denom = (1 + 0.001 * sum_sq) ** 2
        g = 0.5 + sin_term / denom

        result = xp.sum(g, axis=1)

        return result + self.f_global
