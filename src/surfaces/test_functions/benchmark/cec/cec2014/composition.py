# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2014 Composition Functions (F23-F30).

Composition functions create complex landscapes by combining multiple
basic functions with different optima locations.
"""

from typing import Any, Dict, List

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2014 import CEC2014Function


class _CompositionBase(CEC2014Function):
    """Base class for composition functions."""

    _spec = {
        "unimodal": False,
        "separable": False,
    }

    # To be defined by subclasses
    n_functions: int = 0
    sigmas: List[float] = []
    lambdas: List[float] = []
    biases: List[float] = []

    def _get_composition_optima(self) -> np.ndarray:
        """Get optima locations for each component function."""
        data = self._load_data()
        key = f"comp_optima_{self.func_id}"
        if key in data:
            return data[key]
        # Fallback: generate deterministic optima
        rng = np.random.default_rng(seed=self.func_id * 1000)
        return rng.uniform(-80, 80, size=(self.n_functions, self.n_dim))

    def _compute_weights(self, x: np.ndarray, optima: np.ndarray) -> np.ndarray:
        """Compute weights for each component function."""
        weights = np.zeros(self.n_functions)

        for i in range(self.n_functions):
            diff = x - optima[i]
            dist_sq = np.sum(diff**2)
            if dist_sq != 0:
                weights[i] = np.exp(-dist_sq / (2 * self.n_dim * self.sigmas[i] ** 2))
            else:
                weights[i] = 1e10  # Very large weight if at optimum

        # Normalize weights
        max_weight = np.max(weights)
        if max_weight == 0:
            weights = np.ones(self.n_functions) / self.n_functions
        else:
            # Only keep weights that are significant
            for i in range(self.n_functions):
                if weights[i] != max_weight:
                    weights[i] *= 1 - max_weight**10
            weights = weights / np.sum(weights)

        return weights

    def _batch_compute_weights(self, X: ArrayLike, optima: ArrayLike) -> ArrayLike:
        """Compute weights for each component function (batch version).

        Parameters
        ----------
        X : ArrayLike
            Input batch of shape (n_points, n_dim).
        optima : ArrayLike
            Optima locations of shape (n_functions, n_dim).

        Returns
        -------
        ArrayLike
            Weights of shape (n_points, n_functions).
        """
        xp = get_array_namespace(X)

        # X has shape (n_points, n_dim)
        # optima has shape (n_functions, n_dim)
        # diff[i, j, k] = X[i, k] - optima[j, k]
        diff = X[:, None, :] - optima[None, :, :]  # (n_points, n_functions, n_dim)
        dist_sq = xp.sum(diff**2, axis=2)  # (n_points, n_functions)

        # Compute weights: exp(-dist_sq / (2 * n_dim * sigma^2))
        sigmas = xp.asarray(self.sigmas, dtype=X.dtype)
        weights = xp.exp(-dist_sq / (2 * self.n_dim * sigmas**2))

        # Handle case where dist_sq == 0 (at optimum)
        weights = xp.where(dist_sq == 0, 1e10, weights)

        # Normalize weights
        max_weight = xp.max(weights, axis=1, keepdims=True)  # (n_points, 1)

        # Apply transformation: w[i] *= (1 - max_w^10) if w[i] != max_w
        is_max = weights == max_weight
        weights = xp.where(is_max, weights, weights * (1 - max_weight**10))

        # Normalize
        weight_sum = xp.sum(weights, axis=1, keepdims=True)
        weights = xp.where(weight_sum == 0, 1.0 / self.n_functions, weights / weight_sum)

        return weights


# Basic functions for composition (same as hybrid but standalone)
def _sphere(z: np.ndarray) -> float:
    return np.sum(z**2)


def _high_conditioned_elliptic(z: np.ndarray) -> float:
    D = len(z)
    if D == 1:
        return z[0] ** 2
    result = 0.0
    for i in range(D):
        result += (10**6) ** (i / (D - 1)) * z[i] ** 2
    return result


def _bent_cigar(z: np.ndarray) -> float:
    if len(z) == 1:
        return z[0] ** 2
    return z[0] ** 2 + 10**6 * np.sum(z[1:] ** 2)


def _discus(z: np.ndarray) -> float:
    if len(z) == 1:
        return 10**6 * z[0] ** 2
    return 10**6 * z[0] ** 2 + np.sum(z[1:] ** 2)


def _rosenbrock(z: np.ndarray) -> float:
    z = z + 1
    result = 0.0
    for i in range(len(z) - 1):
        result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
    return result


def _ackley(z: np.ndarray) -> float:
    D = len(z)
    sum1 = np.sum(z**2)
    sum2 = np.sum(np.cos(2 * np.pi * z))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / D)) - np.exp(sum2 / D) + 20 + np.e


def _griewank(z: np.ndarray) -> float:
    D = len(z)
    sum_sq = np.sum(z**2) / 4000
    prod_cos = np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1))))
    return sum_sq - prod_cos + 1


def _rastrigin(z: np.ndarray) -> float:
    D = len(z)
    return 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))


def _schwefel(z: np.ndarray) -> float:
    D = len(z)
    z = z + 4.209687462275036e2
    result = 0.0
    for i in range(D):
        zi = z[i]
        if abs(zi) <= 500:
            result += zi * np.sin(np.sqrt(abs(zi)))
        elif zi > 500:
            result += (500 - zi % 500) * np.sin(np.sqrt(abs(500 - zi % 500)))
        else:
            result += (abs(zi) % 500 - 500) * np.sin(np.sqrt(abs(abs(zi) % 500 - 500)))
    return 418.9829 * D - result


def _weierstrass(z: np.ndarray) -> float:
    a, b, k_max = 0.5, 3, 20
    D = len(z)
    result = 0.0
    for i in range(D):
        for k in range(k_max + 1):
            result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))
    offset = sum(a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1))
    return result - D * offset


def _happycat(z: np.ndarray) -> float:
    D = len(z)
    alpha = 1.0 / 8.0
    sum_sq = np.sum(z**2)
    sum_z = np.sum(z)
    return abs(sum_sq - D) ** (2 * alpha) + (0.5 * sum_sq + sum_z) / D + 0.5


def _hgbat(z: np.ndarray) -> float:
    D = len(z)
    sum_sq = np.sum(z**2)
    sum_z = np.sum(z)
    return abs(sum_sq**2 - sum_z**2) ** 0.5 + (0.5 * sum_sq + sum_z) / D + 0.5


def _katsuura(z: np.ndarray) -> float:
    D = len(z)
    result = 1.0
    for i in range(D):
        inner_sum = 0.0
        for j in range(1, 33):
            inner_sum += abs(2**j * z[i] - round(2**j * z[i])) / (2**j)
        result *= (1 + (i + 1) * inner_sum) ** (10 / (D**1.2))
    return (10 / D**2) * result - (10 / D**2)


def _expanded_griewank_rosenbrock(z: np.ndarray) -> float:
    D = len(z)
    z = z + 1
    result = 0.0
    for i in range(D - 1):
        t = 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
        result += t**2 / 4000 - np.cos(t) + 1
    t = 100 * (z[-1] ** 2 - z[0]) ** 2 + (z[-1] - 1) ** 2
    result += t**2 / 4000 - np.cos(t) + 1
    return result


def _expanded_scaffer(z: np.ndarray) -> float:
    D = len(z)

    def schaffer_f6(x1, x2):
        t = x1**2 + x2**2
        return 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2

    result = 0.0
    for i in range(D - 1):
        result += schaffer_f6(z[i], z[i + 1])
    result += schaffer_f6(z[-1], z[0])
    return result


# =========================================================================
# Vectorized basic functions for batch evaluation
# =========================================================================


def _batch_sphere(Z: ArrayLike) -> ArrayLike:
    """Vectorized Sphere: sum(z_i^2)."""
    xp = get_array_namespace(Z)
    return xp.sum(Z**2, axis=1)


def _batch_high_conditioned_elliptic(Z: ArrayLike) -> ArrayLike:
    """Vectorized High Conditioned Elliptic: sum(10^6^(i/(D-1)) * z_i^2)."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    if D == 1:
        return Z[:, 0] ** 2
    i = xp.arange(D, dtype=Z.dtype)
    coeffs = (10**6) ** (i / (D - 1))
    return xp.sum(coeffs * Z**2, axis=1)


def _batch_bent_cigar(Z: ArrayLike) -> ArrayLike:
    """Vectorized Bent Cigar: z_0^2 + 10^6 * sum(z_i^2, i>0)."""
    xp = get_array_namespace(Z)
    if Z.shape[1] == 1:
        return Z[:, 0] ** 2
    return Z[:, 0] ** 2 + 10**6 * xp.sum(Z[:, 1:] ** 2, axis=1)


def _batch_discus(Z: ArrayLike) -> ArrayLike:
    """Vectorized Discus: 10^6 * z_0^2 + sum(z_i^2, i>0)."""
    xp = get_array_namespace(Z)
    if Z.shape[1] == 1:
        return 10**6 * Z[:, 0] ** 2
    return 10**6 * Z[:, 0] ** 2 + xp.sum(Z[:, 1:] ** 2, axis=1)


def _batch_rosenbrock(Z: ArrayLike) -> ArrayLike:
    """Vectorized Rosenbrock: sum(100*(z_i^2 - z_{i+1})^2 + (z_i - 1)^2)."""
    xp = get_array_namespace(Z)
    Z_shifted = Z + 1  # Shift to standard form
    z_i = Z_shifted[:, :-1]
    z_i1 = Z_shifted[:, 1:]
    return xp.sum(100 * (z_i**2 - z_i1) ** 2 + (z_i - 1) ** 2, axis=1)


def _batch_ackley(Z: ArrayLike) -> ArrayLike:
    """Vectorized Ackley function."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    sum1 = xp.sum(Z**2, axis=1)
    sum2 = xp.sum(xp.cos(2 * np.pi * Z), axis=1)
    return -20 * xp.exp(-0.2 * xp.sqrt(sum1 / D)) - xp.exp(sum2 / D) + 20 + np.e


def _batch_griewank(Z: ArrayLike) -> ArrayLike:
    """Vectorized Griewank function."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    sum_sq = xp.sum(Z**2, axis=1) / 4000
    i = xp.arange(1, D + 1, dtype=Z.dtype)
    prod_cos = xp.prod(xp.cos(Z / xp.sqrt(i)), axis=1)
    return sum_sq - prod_cos + 1


def _batch_rastrigin(Z: ArrayLike) -> ArrayLike:
    """Vectorized Rastrigin: 10*D + sum(z_i^2 - 10*cos(2*pi*z_i))."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    return 10 * D + xp.sum(Z**2 - 10 * xp.cos(2 * np.pi * Z), axis=1)


def _batch_schwefel(Z: ArrayLike) -> ArrayLike:
    """Vectorized Schwefel function with boundary handling."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    Z_shifted = Z + 4.209687462275036e2

    abs_z = xp.abs(Z_shifted)

    # Case 1: |z| <= 500
    term1 = Z_shifted * xp.sin(xp.sqrt(abs_z))

    # Case 2: z > 500
    mod_pos = 500 - xp.mod(Z_shifted, 500)
    term2 = mod_pos * xp.sin(xp.sqrt(xp.abs(mod_pos)))

    # Case 3: z < -500
    mod_neg = xp.mod(abs_z, 500) - 500
    term3 = mod_neg * xp.sin(xp.sqrt(xp.abs(mod_neg)))

    result = xp.where(
        abs_z <= 500,
        term1,
        xp.where(Z_shifted > 500, term2, term3),
    )

    return 418.9829 * D - xp.sum(result, axis=1)


def _batch_weierstrass(Z: ArrayLike) -> ArrayLike:
    """Vectorized Weierstrass function."""
    xp = get_array_namespace(Z)
    a, b, k_max = 0.5, 3, 20
    D = Z.shape[1]

    k = xp.arange(k_max + 1, dtype=Z.dtype)
    a_k = a**k
    b_k = b**k

    Z_expanded = Z[:, :, None]
    cos_terms = a_k * xp.cos(2 * np.pi * b_k * (Z_expanded + 0.5))
    result = xp.sum(cos_terms, axis=(1, 2))

    offset_k = a_k * xp.cos(2 * np.pi * b_k * 0.5)
    offset = D * xp.sum(offset_k)

    return result - offset


def _batch_happycat(Z: ArrayLike) -> ArrayLike:
    """Vectorized HappyCat function."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    alpha = 1.0 / 8.0
    sum_sq = xp.sum(Z**2, axis=1)
    sum_z = xp.sum(Z, axis=1)
    return xp.abs(sum_sq - D) ** (2 * alpha) + (0.5 * sum_sq + sum_z) / D + 0.5


def _batch_hgbat(Z: ArrayLike) -> ArrayLike:
    """Vectorized HGBat function."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    sum_sq = xp.sum(Z**2, axis=1)
    sum_z = xp.sum(Z, axis=1)
    return xp.abs(sum_sq**2 - sum_z**2) ** 0.5 + (0.5 * sum_sq + sum_z) / D + 0.5


def _batch_katsuura(Z: ArrayLike) -> ArrayLike:
    """Vectorized Katsuura function."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]

    j = xp.arange(1, 33, dtype=Z.dtype)
    two_j = 2.0**j

    Z_expanded = Z[:, :, None]
    scaled = two_j * Z_expanded
    inner_sum = xp.sum(xp.abs(scaled - xp.round(scaled)) / two_j, axis=2)

    i = xp.arange(1, D + 1, dtype=Z.dtype)
    terms = (1 + i * inner_sum) ** (10 / (D**1.2))

    result = xp.prod(terms, axis=1)
    return (10 / D**2) * result - (10 / D**2)


def _batch_expanded_griewank_rosenbrock(Z: ArrayLike) -> ArrayLike:
    """Vectorized Expanded Griewank-Rosenbrock function."""
    xp = get_array_namespace(Z)
    Z_shifted = Z + 1

    z_i = Z_shifted[:, :-1]
    z_i1 = Z_shifted[:, 1:]

    t_main = 100 * (z_i**2 - z_i1) ** 2 + (z_i - 1) ** 2
    griewank_main = t_main**2 / 4000 - xp.cos(t_main) + 1

    t_wrap = 100 * (Z_shifted[:, -1] ** 2 - Z_shifted[:, 0]) ** 2 + (Z_shifted[:, -1] - 1) ** 2
    griewank_wrap = t_wrap**2 / 4000 - xp.cos(t_wrap) + 1

    return xp.sum(griewank_main, axis=1) + griewank_wrap


def _batch_expanded_scaffer(Z: ArrayLike) -> ArrayLike:
    """Vectorized Expanded Scaffer F6 function."""
    xp = get_array_namespace(Z)

    z_i = Z[:, :-1]
    z_i1 = Z[:, 1:]
    t_main = z_i**2 + z_i1**2
    schaffer_main = 0.5 + (xp.sin(xp.sqrt(t_main)) ** 2 - 0.5) / (1 + 0.001 * t_main) ** 2

    t_wrap = Z[:, -1] ** 2 + Z[:, 0] ** 2
    schaffer_wrap = 0.5 + (xp.sin(xp.sqrt(t_wrap)) ** 2 - 0.5) / (1 + 0.001 * t_wrap) ** 2

    return xp.sum(schaffer_main, axis=1) + schaffer_wrap


class CompositionFunction1(_CompositionBase):
    """F23: Composition Function 1.

    Combines 5 rotated functions with different optima.

    Components:
    1. Rosenbrock
    2. High Conditioned Elliptic
    3. Bent Cigar
    4. Discus
    5. High Conditioned Elliptic

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 5
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [1, 1e-6, 1e-26, 1e-6, 1e-6]
    biases = [0, 100, 200, 300, 400]

    _spec = {
        "name": "Composition Function 1",
        "func_id": 23,
    }

    def _create_objective_function(self) -> None:
        functions = [
            _rosenbrock,
            _high_conditioned_elliptic,
            _bent_cigar,
            _discus,
            _high_conditioned_elliptic,
        ]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            optima = self._get_composition_optima()
            weights = self._compute_weights(x, optima)
            M = self._get_rotation_matrix()

            result = 0.0
            for i in range(self.n_functions):
                z = M @ (x - optima[i])
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F23: Rosenbrock + HCE + Bent Cigar + Discus + HCE."""
        xp = get_array_namespace(X)
        optima = self._get_composition_optima()
        M = self._get_rotation_matrix()

        # Compute weights: (n_points, n_functions)
        weights = self._batch_compute_weights(X, optima)

        # Batch functions list
        batch_funcs = [
            _batch_rosenbrock,
            _batch_high_conditioned_elliptic,
            _batch_bent_cigar,
            _batch_discus,
            _batch_high_conditioned_elliptic,
        ]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            # Z = M @ (X - optima[i]) for each point
            # (X - optima[i]) has shape (n_points, n_dim)
            # M has shape (n_dim, n_dim)
            Z = (X - optima[i]) @ M.T  # Equivalent to M @ (x - o) for each row
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction2(_CompositionBase):
    """F24: Composition Function 2.

    Combines 3 functions with different properties.

    Components:
    1. Schwefel
    2. Rastrigin
    3. High Conditioned Elliptic

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 3
    sigmas = [20, 20, 20]
    lambdas = [10, 1, 1e-6]
    biases = [0, 100, 200]

    _spec = {
        "name": "Composition Function 2",
        "func_id": 24,
    }

    def _create_objective_function(self) -> None:
        functions = [_schwefel, _rastrigin, _high_conditioned_elliptic]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            optima = self._get_composition_optima()
            weights = self._compute_weights(x, optima)
            M = self._get_rotation_matrix()

            result = 0.0
            for i in range(self.n_functions):
                z = M @ (x - optima[i])
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F24: Schwefel + Rastrigin + HCE."""
        xp = get_array_namespace(X)
        optima = self._get_composition_optima()
        M = self._get_rotation_matrix()

        weights = self._batch_compute_weights(X, optima)

        batch_funcs = [_batch_schwefel, _batch_rastrigin, _batch_high_conditioned_elliptic]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction3(_CompositionBase):
    """F25: Composition Function 3.

    Combines 3 functions.

    Components:
    1. Schwefel
    2. Rastrigin
    3. Ackley

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 3
    sigmas = [10, 30, 50]
    lambdas = [10, 1, 1]
    biases = [0, 100, 200]

    _spec = {
        "name": "Composition Function 3",
        "func_id": 25,
    }

    def _create_objective_function(self) -> None:
        functions = [_schwefel, _rastrigin, _ackley]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            optima = self._get_composition_optima()
            weights = self._compute_weights(x, optima)
            M = self._get_rotation_matrix()

            result = 0.0
            for i in range(self.n_functions):
                z = M @ (x - optima[i])
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F25: Schwefel + Rastrigin + Ackley."""
        xp = get_array_namespace(X)
        optima = self._get_composition_optima()
        M = self._get_rotation_matrix()

        weights = self._batch_compute_weights(X, optima)

        batch_funcs = [_batch_schwefel, _batch_rastrigin, _batch_ackley]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction4(_CompositionBase):
    """F26: Composition Function 4.

    Combines 5 functions.

    Components:
    1. Schwefel
    2. HappyCat
    3. High Conditioned Elliptic
    4. Weierstrass
    5. Griewank

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 5
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [10, 1, 1e-6, 1, 1]
    biases = [0, 100, 200, 300, 400]

    _spec = {
        "name": "Composition Function 4",
        "func_id": 26,
    }

    def _create_objective_function(self) -> None:
        functions = [
            _schwefel,
            _happycat,
            _high_conditioned_elliptic,
            _weierstrass,
            _griewank,
        ]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            optima = self._get_composition_optima()
            weights = self._compute_weights(x, optima)
            M = self._get_rotation_matrix()

            result = 0.0
            for i in range(self.n_functions):
                z = M @ (x - optima[i])
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F26: Schwefel + HappyCat + HCE + Weierstrass + Griewank."""
        xp = get_array_namespace(X)
        optima = self._get_composition_optima()
        M = self._get_rotation_matrix()

        weights = self._batch_compute_weights(X, optima)

        batch_funcs = [
            _batch_schwefel,
            _batch_happycat,
            _batch_high_conditioned_elliptic,
            _batch_weierstrass,
            _batch_griewank,
        ]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction5(_CompositionBase):
    """F27: Composition Function 5.

    Combines 5 functions.

    Components:
    1. HGBat
    2. Rastrigin
    3. Schwefel
    4. Weierstrass
    5. High Conditioned Elliptic

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 5
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [10, 10, 2.5, 25, 1e-6]
    biases = [0, 100, 200, 300, 400]

    _spec = {
        "name": "Composition Function 5",
        "func_id": 27,
    }

    def _create_objective_function(self) -> None:
        functions = [
            _hgbat,
            _rastrigin,
            _schwefel,
            _weierstrass,
            _high_conditioned_elliptic,
        ]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            optima = self._get_composition_optima()
            weights = self._compute_weights(x, optima)
            M = self._get_rotation_matrix()

            result = 0.0
            for i in range(self.n_functions):
                z = M @ (x - optima[i])
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F27: HGBat + Rastrigin + Schwefel + Weierstrass + HCE."""
        xp = get_array_namespace(X)
        optima = self._get_composition_optima()
        M = self._get_rotation_matrix()

        weights = self._batch_compute_weights(X, optima)

        batch_funcs = [
            _batch_hgbat,
            _batch_rastrigin,
            _batch_schwefel,
            _batch_weierstrass,
            _batch_high_conditioned_elliptic,
        ]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction6(_CompositionBase):
    """F28: Composition Function 6.

    Combines 5 functions.

    Components:
    1. Expanded Griewank-Rosenbrock
    2. HappyCat
    3. Schwefel
    4. Expanded Scaffer
    5. High Conditioned Elliptic

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 5
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [10, 10, 2.5, 25, 1e-6]
    biases = [0, 100, 200, 300, 400]

    _spec = {
        "name": "Composition Function 6",
        "func_id": 28,
    }

    def _create_objective_function(self) -> None:
        functions = [
            _expanded_griewank_rosenbrock,
            _happycat,
            _schwefel,
            _expanded_scaffer,
            _high_conditioned_elliptic,
        ]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            optima = self._get_composition_optima()
            weights = self._compute_weights(x, optima)
            M = self._get_rotation_matrix()

            result = 0.0
            for i in range(self.n_functions):
                z = M @ (x - optima[i])
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F28: EGR + HappyCat + Schwefel + Expanded Scaffer + HCE."""
        xp = get_array_namespace(X)
        optima = self._get_composition_optima()
        M = self._get_rotation_matrix()

        weights = self._batch_compute_weights(X, optima)

        batch_funcs = [
            _batch_expanded_griewank_rosenbrock,
            _batch_happycat,
            _batch_schwefel,
            _batch_expanded_scaffer,
            _batch_high_conditioned_elliptic,
        ]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction7(_CompositionBase):
    """F29: Composition Function 7.

    Combines 3 hybrid functions.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 3
    sigmas = [10, 30, 50]
    lambdas = [1, 1, 1]
    biases = [0, 100, 200]

    _spec = {
        "name": "Composition Function 7",
        "func_id": 29,
    }

    def _create_objective_function(self) -> None:
        # Hybrid function components
        def hybrid1(z: np.ndarray) -> float:
            # Similar to F17: Elliptic + Bent Cigar + Rastrigin
            D = len(z)
            g1, g2 = D // 3, D // 3
            return (
                _high_conditioned_elliptic(z[:g1])
                + _bent_cigar(z[g1 : g1 + g2])
                + _rastrigin(z[g1 + g2 :])
            )

        def hybrid2(z: np.ndarray) -> float:
            # Similar to F18: Griewank + Weierstrass + Rosenbrock
            D = len(z)
            g1, g2 = D // 3, D // 3
            return _griewank(z[:g1]) + _weierstrass(z[g1 : g1 + g2]) + _rosenbrock(z[g1 + g2 :])

        def hybrid3(z: np.ndarray) -> float:
            # Similar to F19
            D = len(z)
            g1 = D // 4
            g2 = D // 4
            g3 = D // 4
            return (
                _griewank(z[:g1])
                + _weierstrass(z[g1 : g1 + g2])
                + _rosenbrock(z[g1 + g2 : g1 + g2 + g3])
                + _expanded_scaffer(z[g1 + g2 + g3 :])
            )

        functions = [hybrid1, hybrid2, hybrid3]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            optima = self._get_composition_optima()
            weights = self._compute_weights(x, optima)
            M = self._get_rotation_matrix()

            result = 0.0
            for i in range(self.n_functions):
                z = M @ (x - optima[i])
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F29: 3 hybrid functions."""
        xp = get_array_namespace(X)
        optima = self._get_composition_optima()
        M = self._get_rotation_matrix()
        D_rot = M.shape[0]  # Actual dimension after rotation

        weights = self._batch_compute_weights(X, optima)

        # Vectorized hybrid functions (D_rot is used, not self.n_dim)
        def batch_hybrid1(Z: ArrayLike) -> ArrayLike:
            # Elliptic + Bent Cigar + Rastrigin
            g1, g2 = D_rot // 3, D_rot // 3
            return (
                _batch_high_conditioned_elliptic(Z[:, :g1])
                + _batch_bent_cigar(Z[:, g1 : g1 + g2])
                + _batch_rastrigin(Z[:, g1 + g2 :])
            )

        def batch_hybrid2(Z: ArrayLike) -> ArrayLike:
            # Griewank + Weierstrass + Rosenbrock
            g1, g2 = D_rot // 3, D_rot // 3
            return (
                _batch_griewank(Z[:, :g1])
                + _batch_weierstrass(Z[:, g1 : g1 + g2])
                + _batch_rosenbrock(Z[:, g1 + g2 :])
            )

        def batch_hybrid3(Z: ArrayLike) -> ArrayLike:
            # Griewank + Weierstrass + Rosenbrock + Expanded Scaffer
            g1 = D_rot // 4
            g2 = D_rot // 4
            g3 = D_rot // 4
            return (
                _batch_griewank(Z[:, :g1])
                + _batch_weierstrass(Z[:, g1 : g1 + g2])
                + _batch_rosenbrock(Z[:, g1 + g2 : g1 + g2 + g3])
                + _batch_expanded_scaffer(Z[:, g1 + g2 + g3 :])
            )

        batch_funcs = [batch_hybrid1, batch_hybrid2, batch_hybrid3]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction8(_CompositionBase):
    """F30: Composition Function 8.

    Combines 3 hybrid functions (different from F29).

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 3
    sigmas = [10, 30, 50]
    lambdas = [1, 1, 1]
    biases = [0, 100, 200]

    _spec = {
        "name": "Composition Function 8",
        "func_id": 30,
    }

    def _create_objective_function(self) -> None:
        def hybrid4(z: np.ndarray) -> float:
            # Similar to F20
            D = len(z)
            g1 = D // 4
            g2 = D // 4
            g3 = D // 4
            return (
                _hgbat(z[:g1])
                + _discus(z[g1 : g1 + g2])
                + _expanded_griewank_rosenbrock(z[g1 + g2 : g1 + g2 + g3])
                + _rastrigin(z[g1 + g2 + g3 :])
            )

        def hybrid5(z: np.ndarray) -> float:
            # Similar to F21
            D = len(z)
            g = D // 5
            return (
                _expanded_scaffer(z[:g])
                + _hgbat(z[g : 2 * g])
                + _rosenbrock(z[2 * g : 3 * g])
                + _high_conditioned_elliptic(z[3 * g : 4 * g])
                + _ackley(z[4 * g :])
            )

        def hybrid6(z: np.ndarray) -> float:
            # Similar to F22
            D = len(z)
            g = D // 5
            return (
                _katsuura(z[:g])
                + _happycat(z[g : 2 * g])
                + _expanded_griewank_rosenbrock(z[2 * g : 3 * g])
                + _schwefel(z[3 * g : 4 * g])
                + _ackley(z[4 * g :])
            )

        functions = [hybrid4, hybrid5, hybrid6]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            optima = self._get_composition_optima()
            weights = self._compute_weights(x, optima)
            M = self._get_rotation_matrix()

            result = 0.0
            for i in range(self.n_functions):
                z = M @ (x - optima[i])
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F30: 3 hybrid functions."""
        xp = get_array_namespace(X)
        optima = self._get_composition_optima()
        M = self._get_rotation_matrix()
        D_rot = M.shape[0]  # Actual dimension after rotation

        weights = self._batch_compute_weights(X, optima)

        # Vectorized hybrid functions (D_rot is used, not self.n_dim)
        def batch_hybrid4(Z: ArrayLike) -> ArrayLike:
            # HGBat + Discus + EGR + Rastrigin
            g1 = D_rot // 4
            g2 = D_rot // 4
            g3 = D_rot // 4
            return (
                _batch_hgbat(Z[:, :g1])
                + _batch_discus(Z[:, g1 : g1 + g2])
                + _batch_expanded_griewank_rosenbrock(Z[:, g1 + g2 : g1 + g2 + g3])
                + _batch_rastrigin(Z[:, g1 + g2 + g3 :])
            )

        def batch_hybrid5(Z: ArrayLike) -> ArrayLike:
            # Expanded Scaffer + HGBat + Rosenbrock + HCE + Ackley
            g = D_rot // 5
            return (
                _batch_expanded_scaffer(Z[:, :g])
                + _batch_hgbat(Z[:, g : 2 * g])
                + _batch_rosenbrock(Z[:, 2 * g : 3 * g])
                + _batch_high_conditioned_elliptic(Z[:, 3 * g : 4 * g])
                + _batch_ackley(Z[:, 4 * g :])
            )

        def batch_hybrid6(Z: ArrayLike) -> ArrayLike:
            # Katsuura + HappyCat + EGR + Schwefel + Ackley
            g = D_rot // 5
            return (
                _batch_katsuura(Z[:, :g])
                + _batch_happycat(Z[:, g : 2 * g])
                + _batch_expanded_griewank_rosenbrock(Z[:, 2 * g : 3 * g])
                + _batch_schwefel(Z[:, 3 * g : 4 * g])
                + _batch_ackley(Z[:, 4 * g :])
            )

        batch_funcs = [batch_hybrid4, batch_hybrid5, batch_hybrid6]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global
