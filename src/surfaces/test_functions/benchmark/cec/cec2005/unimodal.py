# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2005 Unimodal Functions (F1-F5)."""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2005 import CEC2005Function, CEC2005NonRotatedFunction

# =============================================================================
# F1: Shifted Sphere Function
# =============================================================================


class ShiftedSphere(CEC2005NonRotatedFunction):
    """F1: Shifted Sphere Function.

    f(x) = sum(z_i^2) + f_bias, where z = x - o

    Properties:
    - Unimodal
    - Separable
    - Scalable
    - Bounds: [-100, 100]^D
    """

    _spec = {
        "name": "Shifted Sphere Function",
        "func_id": 1,
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
        "convex": True,
        "separable": True,
    }

    def _create_objective_function(self) -> None:
        def sphere(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            return float(np.sum(z**2) + self.f_global)

        self.pure_objective_function = sphere

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        return xp.sum(Z**2, axis=1) + self.f_global


# =============================================================================
# F2: Shifted Schwefel's Problem 1.2
# =============================================================================


class ShiftedSchwefel12(CEC2005NonRotatedFunction):
    """F2: Shifted Schwefel's Problem 1.2.

    f(x) = sum_i(sum_j=1..i(z_j))^2 + f_bias, where z = x - o

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    - Bounds: [-100, 100]^D
    """

    _spec = {
        "name": "Shifted Schwefel's Problem 1.2",
        "func_id": 2,
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def schwefel_12(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)

            result = 0.0
            for i in range(self.n_dim):
                partial_sum = np.sum(z[: i + 1])
                result += partial_sum**2

            return float(result + self.f_global)

        self.pure_objective_function = schwefel_12

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)

        # Cumulative sum along dimension axis
        cumsum = xp.cumsum(Z, axis=1)
        return xp.sum(cumsum**2, axis=1) + self.f_global


# =============================================================================
# F3: Shifted Rotated High Conditioned Elliptic Function
# =============================================================================


class ShiftedRotatedElliptic(CEC2005Function):
    """F3: Shifted Rotated High Conditioned Elliptic Function.

    f(x) = sum_i(10^6^(i/(D-1)) * z_i^2) + f_bias, where z = M @ (x - o)

    Properties:
    - Unimodal
    - Non-separable (due to rotation)
    - Scalable
    - Bounds: [-100, 100]^D
    - Condition number: 10^6
    """

    _spec = {
        "name": "Shifted Rotated High Conditioned Elliptic Function",
        "func_id": 3,
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def elliptic(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            D = self.n_dim
            result = 0.0
            for i in range(D):
                coeff = (10**6) ** (i / (D - 1)) if D > 1 else 1.0
                result += coeff * z[i] ** 2

            return float(result + self.f_global)

        self.pure_objective_function = elliptic

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        D = self.n_dim
        Z = self._batch_shift_rotate(X)

        i = xp.arange(D, dtype=X.dtype)
        coeffs = xp.power(1e6, i / (D - 1)) if D > 1 else xp.ones(1, dtype=X.dtype)

        return xp.sum(coeffs * Z**2, axis=1) + self.f_global


# =============================================================================
# F4: Shifted Schwefel's Problem 1.2 with Noise in Fitness
# =============================================================================


class ShiftedSchwefel12Noise(CEC2005NonRotatedFunction):
    """F4: Shifted Schwefel's Problem 1.2 with Noise in Fitness.

    f(x) = f_base * (1 + 0.4 * |N(0,1)|) + f_bias
    where f_base = sum_i(sum_j=1..i(z_j))^2, z = x - o

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    - Bounds: [-100, 100]^D
    - Noise: Gaussian N(0,1) multiplicative
    """

    _spec = {
        "name": "Shifted Schwefel's Problem 1.2 with Noise",
        "func_id": 4,
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
        "convex": False,  # Noise breaks convexity
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def schwefel_12_noise(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)

            # Schwefel 1.2 base function
            result = 0.0
            for i in range(self.n_dim):
                partial_sum = np.sum(z[: i + 1])
                result += partial_sum**2

            # Add noise
            noise = np.random.standard_normal()
            result = result * (1 + 0.4 * abs(noise))

            return float(result + self.f_global)

        self.pure_objective_function = schwefel_12_noise

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        Z = self._batch_shift(X, self.func_id)

        # Schwefel 1.2 base function
        cumsum = xp.cumsum(Z, axis=1)
        f_base = xp.sum(cumsum**2, axis=1)

        # Add noise (different for each point)
        noise = xp.asarray(np.random.standard_normal(n_points))
        result = f_base * (1 + 0.4 * xp.abs(noise))

        return result + self.f_global


# =============================================================================
# F5: Schwefel's Problem 2.6 with Global Optimum on Bounds
# =============================================================================


class SchwefelProblem26(CEC2005NonRotatedFunction):
    """F5: Schwefel's Problem 2.6 with Global Optimum on Bounds.

    f(x) = max_i(|A_i * x - B_i|) + f_bias

    where A is a DxD matrix, B = A @ o (o is the global optimum)

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    - Bounds: [-100, 100]^D
    - Global optimum on bounds
    """

    _spec = {
        "name": "Schwefel's Problem 2.6",
        "func_id": 5,
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
        "convex": False,  # max function, not smooth
        "separable": False,
    }

    def _get_a_matrix(self) -> np.ndarray:
        """Get the A matrix for this function."""
        data = self._load_data()
        return data.get("a_matrix_5", np.eye(self.n_dim))

    def _get_b_vector(self) -> np.ndarray:
        """Get the B vector for this function."""
        data = self._load_data()
        return data.get("b_vector_5", np.zeros(self.n_dim))

    def _create_objective_function(self) -> None:
        A = self._get_a_matrix()
        B = self._get_b_vector()

        def schwefel_26(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)

            # Compute |A_i * x - B_i| for each row i
            Ax = A @ x
            diff = np.abs(Ax - B)

            return float(np.max(diff) + self.f_global)

        self.pure_objective_function = schwefel_26

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        A = xp.asarray(self._get_a_matrix())
        B = xp.asarray(self._get_b_vector())

        # X: (n_points, D), A: (D, D), B: (D,)
        # Ax: (n_points, D) = X @ A.T
        Ax = X @ A.T
        diff = xp.abs(Ax - B)

        return xp.max(diff, axis=1) + self.f_global
