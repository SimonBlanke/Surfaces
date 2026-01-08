# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2005 Composition Functions (F15-F25).

These are hybrid composition functions that combine multiple basic functions
with weighted combination based on Gaussian distance to component optima.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2005 import CEC2005Function


# =============================================================================
# Component Functions (basic functions used in compositions)
# =============================================================================


def _rastrigin(z: np.ndarray) -> float:
    """Rastrigin function."""
    return float(np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10))


def _weierstrass(z: np.ndarray, a: float = 0.5, b: int = 3, k_max: int = 20) -> float:
    """Weierstrass function."""
    D = len(z)
    result = 0.0
    const = 0.0
    for k in range(k_max + 1):
        a_k = a**k
        b_k = b**k
        const += a_k * np.cos(np.pi * b_k)
        for i in range(D):
            result += a_k * np.cos(2 * np.pi * b_k * (z[i] + 0.5))
    return float(result - D * const)


def _griewank(z: np.ndarray) -> float:
    """Griewank function."""
    D = len(z)
    sum_term = np.sum(z**2) / 4000
    prod_term = np.prod([np.cos(z[i] / np.sqrt(i + 1)) for i in range(D)])
    return float(sum_term - prod_term + 1)


def _ackley(z: np.ndarray) -> float:
    """Ackley function."""
    D = len(z)
    sum_sq = np.sum(z**2) / D
    sum_cos = np.sum(np.cos(2 * np.pi * z)) / D
    return float(-20 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20 + np.e)


def _sphere(z: np.ndarray) -> float:
    """Sphere function."""
    return float(np.sum(z**2))


def _ef8f2(z: np.ndarray) -> float:
    """Expanded Griewank-Rosenbrock (EF8F2) function."""
    D = len(z)
    result = 0.0
    for i in range(D):
        z_i = z[i]
        z_next = z[(i + 1) % D]
        f2 = 100 * (z_i**2 - z_next) ** 2 + (z_i - 1) ** 2
        result += f2**2 / 4000 - np.cos(f2) + 1
    return float(result)


def _scaffer(z: np.ndarray) -> float:
    """Expanded Scaffer's F6 function."""
    D = len(z)
    result = 0.0
    for i in range(D):
        z_i = z[i]
        z_next = z[(i + 1) % D]
        sum_sq = z_i**2 + z_next**2
        sin_term = np.sin(np.sqrt(sum_sq)) ** 2 - 0.5
        denom = (1 + 0.001 * sum_sq) ** 2
        result += 0.5 + sin_term / denom
    return float(result)


# Batch versions of component functions


def _batch_rastrigin(Z: ArrayLike) -> ArrayLike:
    """Batch Rastrigin."""
    xp = get_array_namespace(Z)
    return xp.sum(Z**2 - 10 * xp.cos(2 * np.pi * Z) + 10, axis=1)


def _batch_weierstrass(
    Z: ArrayLike, a: float = 0.5, b: int = 3, k_max: int = 20
) -> ArrayLike:
    """Batch Weierstrass."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]

    k = xp.arange(k_max + 1, dtype=Z.dtype)
    a_k = xp.power(a, k)
    b_k = xp.power(float(b), k)

    # const = sum_k(a^k * cos(pi * b^k))
    const = xp.sum(a_k * xp.cos(np.pi * b_k))

    # result = sum_i sum_k(a^k * cos(2*pi*b^k*(z_i+0.5)))
    Z_expanded = Z[:, :, None] + 0.5
    terms = a_k * xp.cos(2 * np.pi * b_k * Z_expanded)
    result = xp.sum(xp.sum(terms, axis=2), axis=1)

    return result - D * const


def _batch_griewank(Z: ArrayLike) -> ArrayLike:
    """Batch Griewank."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    sum_term = xp.sum(Z**2, axis=1) / 4000
    sqrt_i = xp.sqrt(xp.arange(1, D + 1, dtype=Z.dtype))
    prod_term = xp.prod(xp.cos(Z / sqrt_i), axis=1)
    return sum_term - prod_term + 1


def _batch_ackley(Z: ArrayLike) -> ArrayLike:
    """Batch Ackley."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    sum_sq = xp.sum(Z**2, axis=1) / D
    sum_cos = xp.sum(xp.cos(2 * np.pi * Z), axis=1) / D
    return -20 * xp.exp(-0.2 * xp.sqrt(sum_sq)) - xp.exp(sum_cos) + 20 + np.e


def _batch_sphere(Z: ArrayLike) -> ArrayLike:
    """Batch Sphere."""
    xp = get_array_namespace(Z)
    return xp.sum(Z**2, axis=1)


def _batch_ef8f2(Z: ArrayLike) -> ArrayLike:
    """Batch EF8F2."""
    xp = get_array_namespace(Z)
    Z_next = xp.roll(Z, -1, axis=1)
    F2 = 100 * (Z**2 - Z_next) ** 2 + (Z - 1) ** 2
    return xp.sum(F2**2 / 4000 - xp.cos(F2) + 1, axis=1)


def _batch_scaffer(Z: ArrayLike) -> ArrayLike:
    """Batch Scaffer F6."""
    xp = get_array_namespace(Z)
    Z_next = xp.roll(Z, -1, axis=1)
    sum_sq = Z**2 + Z_next**2
    sin_term = xp.sin(xp.sqrt(sum_sq)) ** 2 - 0.5
    denom = (1 + 0.001 * sum_sq) ** 2
    return xp.sum(0.5 + sin_term / denom, axis=1)


# =============================================================================
# Base Composition Class
# =============================================================================


class _CompositionBase(CEC2005Function):
    """Base class for CEC 2005 composition functions.

    Composition functions combine multiple basic functions with
    weighted combination based on Gaussian distance to component optima.
    """

    n_functions: int = 10
    sigmas: List[float] = []
    lambdas: List[float] = []
    component_funcs: List[Callable] = []
    batch_component_funcs: List[Callable] = []
    use_rotation: bool = True

    _spec = {
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "convex": False,
        "separable": False,
    }

    @property
    def x_global(self) -> Optional[np.ndarray]:
        """Global optimum location (first composition optimum)."""
        optima = self._get_composition_optima()
        return optima[0] if optima is not None else None

    def _get_composition_optima(self) -> np.ndarray:
        """Get optima for each component function."""
        data = self._load_data()
        key = f"comp_optima_{self.func_id}"
        return data.get(key, np.zeros((self.n_functions, self.n_dim)))

    def _get_component_rotation(self, index: int) -> np.ndarray:
        """Get rotation matrix for a specific component function."""
        if not self.use_rotation:
            return np.eye(self.n_dim)

        data = self._load_data()
        key = f"rotation_{self.func_id}_{index + 1}"
        return data.get(key, np.eye(self.n_dim))

    def _compute_weights(
        self, x: np.ndarray, optima: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Compute weights for each component function.

        Returns
        -------
        weights : np.ndarray
            Normalized weights for each component.
        max_idx : int
            Index of component with maximum weight.
        """
        # Compute distances to each optimum
        diff = x - optima  # (n_functions, D)
        dist_sq = np.sum(diff**2, axis=1)  # (n_functions,)

        # Gaussian weights
        sigmas = np.array(self.sigmas)
        weights = np.exp(-dist_sq / (2 * self.n_dim * sigmas**2))

        # Find max weight
        max_idx = np.argmax(weights)
        max_weight = weights[max_idx]

        # Modify weights based on max
        weights = np.where(
            np.arange(self.n_functions) == max_idx,
            weights,
            weights * (1 - max_weight**10),
        )

        # Normalize
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(self.n_functions) / self.n_functions

        return weights, max_idx

    def _create_objective_function(self) -> None:
        optima = self._get_composition_optima()
        lambdas = np.array(self.lambdas)

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)

            # Compute weights
            weights, _ = self._compute_weights(x, optima)

            result = 0.0
            for i in range(self.n_functions):
                # Transform: z = M @ (x - o_i) * lambda_i
                z = x - optima[i]
                if self.use_rotation:
                    M = self._get_component_rotation(i)
                    z = M @ z
                z = z * lambdas[i]

                # Evaluate component function
                f_i = self.component_funcs[i](z)

                # Add weighted contribution
                result += weights[i] * f_i

            return float(result + self.f_global)

        self.pure_objective_function = composition

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        optima = xp.asarray(self._get_composition_optima())
        lambdas = xp.asarray(self.lambdas, dtype=X.dtype)
        sigmas = xp.asarray(self.sigmas, dtype=X.dtype)

        # Compute weights for all points
        # X: (n_points, D), optima: (n_functions, D)
        diff = X[:, None, :] - optima[None, :, :]  # (n_points, n_functions, D)
        dist_sq = xp.sum(diff**2, axis=2)  # (n_points, n_functions)

        weights = xp.exp(-dist_sq / (2 * self.n_dim * sigmas**2))

        # Find max weight per point
        max_weight = xp.max(weights, axis=1, keepdims=True)
        is_max = weights == max_weight

        # Modify weights
        weights = xp.where(is_max, weights, weights * (1 - max_weight**10))

        # Normalize
        weight_sum = xp.sum(weights, axis=1, keepdims=True)
        weights = xp.where(
            weight_sum > 0, weights / weight_sum, 1.0 / self.n_functions
        )

        # Evaluate each component and accumulate
        result = xp.zeros(n_points, dtype=X.dtype)

        for i in range(self.n_functions):
            # Transform: Z = (X - o_i) @ M.T * lambda_i
            Z = X - optima[i]
            if self.use_rotation:
                M = xp.asarray(self._get_component_rotation(i))
                Z = Z @ M.T
            Z = Z * lambdas[i]

            # Evaluate component
            f_i = self.batch_component_funcs[i](Z)

            # Weighted sum
            result = result + weights[:, i] * f_i

        return result + self.f_global


# =============================================================================
# F15: Rotated Hybrid Composition Function 1
# =============================================================================


class CompositionFunction1(_CompositionBase):
    """F15: Rotated Hybrid Composition Function 1.

    Combines: Rastrigin, Rastrigin, Weierstrass, Weierstrass, Griewank,
              Griewank, Ackley, Ackley, Sphere, Sphere

    Properties:
    - Multimodal
    - Non-separable
    - Bounds: [-5, 5]^D
    """

    _spec = {
        "name": "Rotated Hybrid Composition Function 1",
        "func_id": 15,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "separable": False,
    }

    n_functions = 10
    sigmas = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    lambdas = [1, 1, 10, 10, 5.0 / 60, 5.0 / 60, 5.0 / 32, 5.0 / 32, 5.0 / 100, 5.0 / 100]
    component_funcs = [
        _rastrigin,
        _rastrigin,
        _weierstrass,
        _weierstrass,
        _griewank,
        _griewank,
        _ackley,
        _ackley,
        _sphere,
        _sphere,
    ]
    batch_component_funcs = [
        _batch_rastrigin,
        _batch_rastrigin,
        _batch_weierstrass,
        _batch_weierstrass,
        _batch_griewank,
        _batch_griewank,
        _batch_ackley,
        _batch_ackley,
        _batch_sphere,
        _batch_sphere,
    ]
    use_rotation = False


# =============================================================================
# F16: Rotated Hybrid Composition Function 2
# =============================================================================


class CompositionFunction2(_CompositionBase):
    """F16: Rotated Hybrid Composition Function 2.

    Same as F15 but with rotation matrices.
    """

    _spec = {
        "name": "Rotated Hybrid Composition Function 2",
        "func_id": 16,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "separable": False,
    }

    n_functions = 10
    sigmas = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    lambdas = [1, 1, 10, 10, 5.0 / 60, 5.0 / 60, 5.0 / 32, 5.0 / 32, 5.0 / 100, 5.0 / 100]
    component_funcs = [
        _rastrigin,
        _rastrigin,
        _weierstrass,
        _weierstrass,
        _griewank,
        _griewank,
        _ackley,
        _ackley,
        _sphere,
        _sphere,
    ]
    batch_component_funcs = [
        _batch_rastrigin,
        _batch_rastrigin,
        _batch_weierstrass,
        _batch_weierstrass,
        _batch_griewank,
        _batch_griewank,
        _batch_ackley,
        _batch_ackley,
        _batch_sphere,
        _batch_sphere,
    ]
    use_rotation = True


# =============================================================================
# F17: Rotated Hybrid Composition Function 3 (with Noise)
# =============================================================================


class CompositionFunction3(CompositionFunction2):
    """F17: Rotated Hybrid Composition Function 3 with Noise in Fitness.

    Same as F16 but with Gaussian noise added to fitness.
    """

    _spec = {
        "name": "Rotated Hybrid Composition Function 3 with Noise",
        "func_id": 17,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        super()._create_objective_function()
        base_func = self.pure_objective_function

        def composition_noise(params: Dict[str, Any]) -> float:
            result = base_func(params)
            # Add noise: f * (1 + 0.2 * |N(0,1)|)
            noise = np.random.standard_normal()
            return float((result - self.f_global) * (1 + 0.2 * abs(noise)) + self.f_global)

        self.pure_objective_function = composition_noise

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        result = super()._batch_objective(X)
        # Add noise
        noise = xp.asarray(np.random.standard_normal(X.shape[0]))
        return (result - self.f_global) * (1 + 0.2 * xp.abs(noise)) + self.f_global


# =============================================================================
# F18: Rotated Hybrid Composition Function 4
# =============================================================================


class CompositionFunction4(_CompositionBase):
    """F18: Rotated Hybrid Composition Function 4.

    With narrower basin for global optimum.
    """

    _spec = {
        "name": "Rotated Hybrid Composition Function 4",
        "func_id": 18,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "separable": False,
    }

    n_functions = 10
    sigmas = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    lambdas = [
        10,
        10,
        10,
        10,
        5.0 / 60,
        5.0 / 60,
        5.0 / 32,
        5.0 / 32,
        5.0 / 100,
        5.0 / 100,
    ]
    component_funcs = [
        _ackley,
        _ackley,
        _rastrigin,
        _rastrigin,
        _sphere,
        _sphere,
        _weierstrass,
        _weierstrass,
        _griewank,
        _griewank,
    ]
    batch_component_funcs = [
        _batch_ackley,
        _batch_ackley,
        _batch_rastrigin,
        _batch_rastrigin,
        _batch_sphere,
        _batch_sphere,
        _batch_weierstrass,
        _batch_weierstrass,
        _batch_griewank,
        _batch_griewank,
    ]
    use_rotation = True


# =============================================================================
# F19: Rotated Hybrid Composition Function 5 (with Bounds)
# =============================================================================


class CompositionFunction5(CompositionFunction4):
    """F19: Rotated Hybrid Composition Function 5.

    F18 with a narrow basin for global optimum.
    """

    _spec = {
        "name": "Rotated Hybrid Composition Function 5",
        "func_id": 19,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "separable": False,
    }

    # Narrower sigmas for first component
    sigmas = [0.1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2]


# =============================================================================
# F20: Rotated Hybrid Composition Function 6 (Global Optimum on Bounds)
# =============================================================================


class CompositionFunction6(CompositionFunction5):
    """F20: Rotated Hybrid Composition Function 6.

    F19 with global optimum on bounds.
    """

    _spec = {
        "name": "Rotated Hybrid Composition Function 6",
        "func_id": 20,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "separable": False,
    }


# =============================================================================
# F21: Rotated Hybrid Composition Function 7
# =============================================================================


class CompositionFunction7(_CompositionBase):
    """F21: Rotated Hybrid Composition Function 7.

    Different component functions: EF8F2 (Griewank-Rosenbrock), Scaffer.
    """

    _spec = {
        "name": "Rotated Hybrid Composition Function 7",
        "func_id": 21,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "separable": False,
    }

    n_functions = 10
    sigmas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    lambdas = [
        1.0 / 5,
        1.0 / 5,
        1,
        1,
        5.0 / 60,
        5.0 / 60,
        5.0 / 32,
        5.0 / 32,
        5.0 / 100,
        5.0 / 100,
    ]
    component_funcs = [
        _ef8f2,
        _ef8f2,
        _scaffer,
        _scaffer,
        _griewank,
        _griewank,
        _rastrigin,
        _rastrigin,
        _weierstrass,
        _weierstrass,
    ]
    batch_component_funcs = [
        _batch_ef8f2,
        _batch_ef8f2,
        _batch_scaffer,
        _batch_scaffer,
        _batch_griewank,
        _batch_griewank,
        _batch_rastrigin,
        _batch_rastrigin,
        _batch_weierstrass,
        _batch_weierstrass,
    ]
    use_rotation = True


# =============================================================================
# F22: Rotated Hybrid Composition Function 8
# =============================================================================


class CompositionFunction8(CompositionFunction7):
    """F22: Rotated Hybrid Composition Function 8.

    F21 with narrower basin for global optimum.
    """

    _spec = {
        "name": "Rotated Hybrid Composition Function 8",
        "func_id": 22,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "separable": False,
    }

    # Narrower sigmas
    sigmas = [0.1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2]


# =============================================================================
# F23: Rotated Hybrid Composition Function 9 (Non-Continuous)
# =============================================================================


class CompositionFunction9(CompositionFunction7):
    """F23: Non-Continuous Rotated Hybrid Composition Function 9.

    F21 with non-continuous transformation.
    """

    _spec = {
        "name": "Non-Continuous Rotated Hybrid Composition Function",
        "func_id": 23,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "separable": False,
    }

    def _non_continuous_transform(self, x: np.ndarray, o: np.ndarray) -> np.ndarray:
        """Apply non-continuous transformation."""
        y = x.copy()
        for i in range(len(x)):
            if abs(x[i] - o[i]) >= 0.5:
                y[i] = round(2 * x[i]) / 2
        return y

    def _create_objective_function(self) -> None:
        optima = self._get_composition_optima()
        lambdas = np.array(self.lambdas)

        def composition_nc(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)

            # Apply non-continuous transform relative to first optimum
            x_nc = self._non_continuous_transform(x, optima[0])

            # Compute weights
            weights, _ = self._compute_weights(x_nc, optima)

            result = 0.0
            for i in range(self.n_functions):
                z = x_nc - optima[i]
                if self.use_rotation:
                    M = self._get_component_rotation(i)
                    z = M @ z
                z = z * lambdas[i]
                f_i = self.component_funcs[i](z)
                result += weights[i] * f_i

            return float(result + self.f_global)

        self.pure_objective_function = composition_nc

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Batch evaluation with non-continuous transform."""
        xp = get_array_namespace(X)
        optima = xp.asarray(self._get_composition_optima())

        # Non-continuous transform
        diff = xp.abs(X - optima[0])
        X_nc = xp.where(diff >= 0.5, xp.round(2 * X) / 2, X)

        # Use parent batch implementation with transformed input
        # We need to temporarily set X to X_nc
        n_points = X.shape[0]
        lambdas = xp.asarray(self.lambdas, dtype=X.dtype)
        sigmas = xp.asarray(self.sigmas, dtype=X.dtype)

        # Compute weights
        diff = X_nc[:, None, :] - optima[None, :, :]
        dist_sq = xp.sum(diff**2, axis=2)
        weights = xp.exp(-dist_sq / (2 * self.n_dim * sigmas**2))
        max_weight = xp.max(weights, axis=1, keepdims=True)
        is_max = weights == max_weight
        weights = xp.where(is_max, weights, weights * (1 - max_weight**10))
        weight_sum = xp.sum(weights, axis=1, keepdims=True)
        weights = xp.where(weight_sum > 0, weights / weight_sum, 1.0 / self.n_functions)

        result = xp.zeros(n_points, dtype=X.dtype)
        for i in range(self.n_functions):
            Z = X_nc - optima[i]
            if self.use_rotation:
                M = xp.asarray(self._get_component_rotation(i))
                Z = Z @ M.T
            Z = Z * lambdas[i]
            f_i = self.batch_component_funcs[i](Z)
            result = result + weights[:, i] * f_i

        return result + self.f_global


# =============================================================================
# F24: Rotated Hybrid Composition Function 10
# =============================================================================


class CompositionFunction10(_CompositionBase):
    """F24: Rotated Hybrid Composition Function 10.

    Different sigma and lambda values.
    """

    _spec = {
        "name": "Rotated Hybrid Composition Function 10",
        "func_id": 24,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "separable": False,
    }

    n_functions = 10
    sigmas = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    lambdas = [10, 10, 2.5, 25, 1e-6, 1e-6, 2.5e-4, 2.5e-4, 2.5e-4, 2.5e-4]
    component_funcs = [
        _weierstrass,
        _scaffer,
        _weierstrass,
        _scaffer,
        _sphere,
        _sphere,
        _griewank,
        _griewank,
        _rastrigin,
        _rastrigin,
    ]
    batch_component_funcs = [
        _batch_weierstrass,
        _batch_scaffer,
        _batch_weierstrass,
        _batch_scaffer,
        _batch_sphere,
        _batch_sphere,
        _batch_griewank,
        _batch_griewank,
        _batch_rastrigin,
        _batch_rastrigin,
    ]
    use_rotation = True


# =============================================================================
# F25: Rotated Hybrid Composition Function 11 (Global Optimum on Bounds)
# =============================================================================


class CompositionFunction11(CompositionFunction10):
    """F25: Rotated Hybrid Composition Function 11.

    F24 with global optimum on bounds.
    """

    _spec = {
        "name": "Rotated Hybrid Composition Function 11",
        "func_id": 25,
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
        "separable": False,
    }
