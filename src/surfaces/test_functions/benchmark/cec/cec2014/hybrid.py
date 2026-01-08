# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2014 Hybrid Functions (F17-F22).

Hybrid functions divide the variables into groups and apply different
basic functions to each group.
"""

from typing import Any, Dict, List

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2014 import CEC2014Function


class _HybridBase(CEC2014Function):
    """Base class for hybrid functions."""

    _spec = {
        "unimodal": False,
        "separable": False,
    }

    # To be defined by subclasses
    n_functions: int = 0
    proportions: List[float] = []

    def _get_group_sizes(self) -> List[int]:
        """Calculate group sizes based on proportions."""
        sizes = []
        remaining = self.n_dim
        cumsum = 0
        for i, p in enumerate(self.proportions[:-1]):
            cumsum += p
            size = max(1, int(np.round(self.n_dim * p)))
            sizes.append(size)
            remaining -= size
        sizes.append(remaining)  # Last group gets remainder
        return sizes

    def _split_variables(self, z: np.ndarray) -> List[np.ndarray]:
        """Split variables into groups after shuffling."""
        shuffle_idx = self._get_shuffle_indices()
        z_shuffled = z[shuffle_idx]

        groups = []
        start = 0
        for size in self._get_group_sizes():
            groups.append(z_shuffled[start : start + size])
            start += size
        return groups

    def _batch_split_variables(self, Z: ArrayLike) -> List[ArrayLike]:
        """Split variables into groups after shuffling (batch version).

        Parameters
        ----------
        Z : ArrayLike
            Transformed batch of shape (n_points, n_dim).

        Returns
        -------
        List[ArrayLike]
            List of arrays, each of shape (n_points, group_size).
        """
        shuffle_idx = self._get_shuffle_indices()
        Z_shuffled = Z[:, shuffle_idx]

        groups = []
        start = 0
        for size in self._get_group_sizes():
            groups.append(Z_shuffled[:, start : start + size])
            start += size
        return groups


# Basic functions used in hybrids
def _high_conditioned_elliptic(z: np.ndarray) -> float:
    D = len(z)
    if D == 1:
        return z[0] ** 2
    result = 0.0
    for i in range(D):
        result += (10**6) ** (i / (D - 1)) * z[i] ** 2
    return result


def _bent_cigar(z: np.ndarray) -> float:
    return z[0] ** 2 + 10**6 * np.sum(z[1:] ** 2)


def _discus(z: np.ndarray) -> float:
    return 10**6 * z[0] ** 2 + np.sum(z[1:] ** 2)


def _rosenbrock(z: np.ndarray) -> float:
    z = z + 1  # Shift to standard form
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


def _katsuura(z: np.ndarray) -> float:
    D = len(z)
    result = 1.0
    for i in range(D):
        inner_sum = 0.0
        for j in range(1, 33):
            inner_sum += abs(2**j * z[i] - round(2**j * z[i])) / (2**j)
        result *= (1 + (i + 1) * inner_sum) ** (10 / (D**1.2))
    return (10 / D**2) * result - (10 / D**2)


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
    return Z[:, 0] ** 2 + 10**6 * xp.sum(Z[:, 1:] ** 2, axis=1)


def _batch_discus(Z: ArrayLike) -> ArrayLike:
    """Vectorized Discus: 10^6 * z_0^2 + sum(z_i^2, i>0)."""
    xp = get_array_namespace(Z)
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
    # prod(cos(z_i / sqrt(i+1)))
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

    # Handle three cases: |z| <= 500, z > 500, z < -500
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

    # Create k values: shape (k_max+1,)
    k = xp.arange(k_max + 1, dtype=Z.dtype)
    a_k = a**k  # shape (k_max+1,)
    b_k = b**k  # shape (k_max+1,)

    # Z has shape (n_points, D)
    # We need sum over i in D, sum over k in k_max+1
    # Z[:, :, None] has shape (n_points, D, 1)
    # a_k * cos(2*pi*b_k*(z+0.5)) for each z
    Z_expanded = Z[:, :, None]  # (n_points, D, 1)
    cos_terms = a_k * xp.cos(2 * np.pi * b_k * (Z_expanded + 0.5))  # (n_points, D, k_max+1)
    result = xp.sum(cos_terms, axis=(1, 2))  # Sum over D and k

    # Offset: D * sum_k(a^k * cos(2*pi*b^k*0.5))
    offset_k = a_k * xp.cos(2 * np.pi * b_k * 0.5)
    offset = D * xp.sum(offset_k)

    return result - offset


def _batch_katsuura(Z: ArrayLike) -> ArrayLike:
    """Vectorized Katsuura function."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]

    # j values: 1 to 32
    j = xp.arange(1, 33, dtype=Z.dtype)
    two_j = 2.0**j  # (32,)

    # Z has shape (n_points, D)
    # For each z_i, compute sum_j |2^j * z_i - round(2^j * z_i)| / 2^j
    Z_expanded = Z[:, :, None]  # (n_points, D, 1)
    scaled = two_j * Z_expanded  # (n_points, D, 32)
    inner_sum = xp.sum(xp.abs(scaled - xp.round(scaled)) / two_j, axis=2)  # (n_points, D)

    # (1 + (i+1) * inner_sum)^(10/D^1.2)
    i = xp.arange(1, D + 1, dtype=Z.dtype)  # (D,)
    terms = (1 + i * inner_sum) ** (10 / (D**1.2))  # (n_points, D)

    result = xp.prod(terms, axis=1)  # (n_points,)
    return (10 / D**2) * result - (10 / D**2)


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


def _batch_expanded_griewank_rosenbrock(Z: ArrayLike) -> ArrayLike:
    """Vectorized Expanded Griewank-Rosenbrock function."""
    xp = get_array_namespace(Z)
    Z_shifted = Z + 1

    # Pairs (z_i, z_{i+1}) for i = 0..D-2, plus (z_{D-1}, z_0)
    z_i = Z_shifted[:, :-1]  # (n_points, D-1)
    z_i1 = Z_shifted[:, 1:]  # (n_points, D-1)

    # t = 100*(z_i^2 - z_{i+1})^2 + (z_i - 1)^2
    t_main = 100 * (z_i**2 - z_i1) ** 2 + (z_i - 1) ** 2
    griewank_main = t_main**2 / 4000 - xp.cos(t_main) + 1

    # Wrap-around term: (z_{D-1}, z_0)
    t_wrap = 100 * (Z_shifted[:, -1] ** 2 - Z_shifted[:, 0]) ** 2 + (Z_shifted[:, -1] - 1) ** 2
    griewank_wrap = t_wrap**2 / 4000 - xp.cos(t_wrap) + 1

    return xp.sum(griewank_main, axis=1) + griewank_wrap


def _batch_expanded_scaffer(Z: ArrayLike) -> ArrayLike:
    """Vectorized Expanded Scaffer F6 function."""
    xp = get_array_namespace(Z)

    # Pairs (z_i, z_{i+1}) for i = 0..D-2
    z_i = Z[:, :-1]
    z_i1 = Z[:, 1:]
    t_main = z_i**2 + z_i1**2
    schaffer_main = 0.5 + (xp.sin(xp.sqrt(t_main)) ** 2 - 0.5) / (1 + 0.001 * t_main) ** 2

    # Wrap-around term: (z_{D-1}, z_0)
    t_wrap = Z[:, -1] ** 2 + Z[:, 0] ** 2
    schaffer_wrap = 0.5 + (xp.sin(xp.sqrt(t_wrap)) ** 2 - 0.5) / (1 + 0.001 * t_wrap) ** 2

    return xp.sum(schaffer_main, axis=1) + schaffer_wrap


class HybridFunction1(_HybridBase):
    """F17: Hybrid Function 1.

    Combines: High Conditioned Elliptic, Bent Cigar, Rastrigin.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 3
    proportions = [0.3, 0.3, 0.4]

    _spec = {
        "name": "Hybrid Function 1",
        "func_id": 17,
    }

    def _create_objective_function(self) -> None:
        functions = [_high_conditioned_elliptic, _bent_cigar, _rastrigin]

        def hybrid(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            groups = self._split_variables(z)

            result = 0.0
            for i, (group, func) in enumerate(zip(groups, functions)):
                if len(group) > 0:
                    result += func(group)

            return result + self.f_global

        self.pure_objective_function = hybrid

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F17: High Conditioned Elliptic + Bent Cigar + Rastrigin."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        groups = self._batch_split_variables(Z)

        batch_funcs = [_batch_high_conditioned_elliptic, _batch_bent_cigar, _batch_rastrigin]
        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for group, func in zip(groups, batch_funcs):
            if group.shape[1] > 0:
                result = result + func(group)

        return result + self.f_global


class HybridFunction2(_HybridBase):
    """F18: Hybrid Function 2.

    Combines: Griewank, Weierstrass, Rosenbrock.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 3
    proportions = [0.3, 0.3, 0.4]

    _spec = {
        "name": "Hybrid Function 2",
        "func_id": 18,
    }

    def _create_objective_function(self) -> None:
        def weierstrass(z):
            a, b, k_max = 0.5, 3, 20
            D = len(z)
            result = 0.0
            for i in range(D):
                for k in range(k_max + 1):
                    result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))
            offset = sum(a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1))
            return result - D * offset

        functions = [_griewank, weierstrass, _rosenbrock]

        def hybrid(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            groups = self._split_variables(z)

            result = 0.0
            for group, func in zip(groups, functions):
                if len(group) > 0:
                    result += func(group)

            return result + self.f_global

        self.pure_objective_function = hybrid

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F18: Griewank + Weierstrass + Rosenbrock."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        groups = self._batch_split_variables(Z)

        batch_funcs = [_batch_griewank, _batch_weierstrass, _batch_rosenbrock]
        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for group, func in zip(groups, batch_funcs):
            if group.shape[1] > 0:
                result = result + func(group)

        return result + self.f_global


class HybridFunction3(_HybridBase):
    """F19: Hybrid Function 3.

    Combines: Griewank, Weierstrass, Rosenbrock, Expanded Scaffer.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 4
    proportions = [0.2, 0.2, 0.3, 0.3]

    _spec = {
        "name": "Hybrid Function 3",
        "func_id": 19,
    }

    def _create_objective_function(self) -> None:
        def weierstrass(z):
            a, b, k_max = 0.5, 3, 20
            D = len(z)
            result = 0.0
            for i in range(D):
                for k in range(k_max + 1):
                    result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))
            offset = sum(a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1))
            return result - D * offset

        functions = [_griewank, weierstrass, _rosenbrock, _expanded_scaffer]

        def hybrid(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            groups = self._split_variables(z)

            result = 0.0
            for group, func in zip(groups, functions):
                if len(group) > 0:
                    result += func(group)

            return result + self.f_global

        self.pure_objective_function = hybrid

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F19: Griewank + Weierstrass + Rosenbrock + Expanded Scaffer."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        groups = self._batch_split_variables(Z)

        batch_funcs = [
            _batch_griewank,
            _batch_weierstrass,
            _batch_rosenbrock,
            _batch_expanded_scaffer,
        ]
        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for group, func in zip(groups, batch_funcs):
            if group.shape[1] > 0:
                result = result + func(group)

        return result + self.f_global


class HybridFunction4(_HybridBase):
    """F20: Hybrid Function 4.

    Combines: HGBat, Discus, Expanded Griewank-Rosenbrock, Rastrigin.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 4
    proportions = [0.2, 0.2, 0.3, 0.3]

    _spec = {
        "name": "Hybrid Function 4",
        "func_id": 20,
    }

    def _create_objective_function(self) -> None:
        functions = [_hgbat, _discus, _expanded_griewank_rosenbrock, _rastrigin]

        def hybrid(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            groups = self._split_variables(z)

            result = 0.0
            for group, func in zip(groups, functions):
                if len(group) > 0:
                    result += func(group)

            return result + self.f_global

        self.pure_objective_function = hybrid

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F20: HGBat + Discus + Expanded Griewank-Rosenbrock + Rastrigin."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        groups = self._batch_split_variables(Z)

        batch_funcs = [
            _batch_hgbat,
            _batch_discus,
            _batch_expanded_griewank_rosenbrock,
            _batch_rastrigin,
        ]
        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for group, func in zip(groups, batch_funcs):
            if group.shape[1] > 0:
                result = result + func(group)

        return result + self.f_global


class HybridFunction5(_HybridBase):
    """F21: Hybrid Function 5.

    Combines: Expanded Scaffer, HGBat, Rosenbrock, High Conditioned Elliptic.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 5
    proportions = [0.1, 0.2, 0.2, 0.2, 0.3]

    _spec = {
        "name": "Hybrid Function 5",
        "func_id": 21,
    }

    def _create_objective_function(self) -> None:
        functions = [
            _expanded_scaffer,
            _hgbat,
            _rosenbrock,
            _high_conditioned_elliptic,
            _ackley,
        ]

        def hybrid(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            groups = self._split_variables(z)

            result = 0.0
            for group, func in zip(groups, functions):
                if len(group) > 0:
                    result += func(group)

            return result + self.f_global

        self.pure_objective_function = hybrid

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F21: Expanded Scaffer + HGBat + Rosenbrock + HCE + Ackley."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        groups = self._batch_split_variables(Z)

        batch_funcs = [
            _batch_expanded_scaffer,
            _batch_hgbat,
            _batch_rosenbrock,
            _batch_high_conditioned_elliptic,
            _batch_ackley,
        ]
        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for group, func in zip(groups, batch_funcs):
            if group.shape[1] > 0:
                result = result + func(group)

        return result + self.f_global


class HybridFunction6(_HybridBase):
    """F22: Hybrid Function 6.

    Combines: Katsuura, HappyCat, Expanded Griewank-Rosenbrock, Schwefel, Ackley.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 5
    proportions = [0.1, 0.2, 0.2, 0.2, 0.3]

    _spec = {
        "name": "Hybrid Function 6",
        "func_id": 22,
    }

    def _create_objective_function(self) -> None:
        functions = [
            _katsuura,
            _happycat,
            _expanded_griewank_rosenbrock,
            _schwefel,
            _ackley,
        ]

        def hybrid(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            groups = self._split_variables(z)

            result = 0.0
            for group, func in zip(groups, functions):
                if len(group) > 0:
                    result += func(group)

            return result + self.f_global

        self.pure_objective_function = hybrid

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F22: Katsuura + HappyCat + EGR + Schwefel + Ackley."""
        xp = get_array_namespace(X)
        Z = self._batch_shift_rotate(X)
        groups = self._batch_split_variables(Z)

        batch_funcs = [
            _batch_katsuura,
            _batch_happycat,
            _batch_expanded_griewank_rosenbrock,
            _batch_schwefel,
            _batch_ackley,
        ]
        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for group, func in zip(groups, batch_funcs):
            if group.shape[1] > 0:
                result = result + func(group)

        return result + self.f_global
