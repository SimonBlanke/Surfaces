# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2014 Hybrid Functions (F17-F22).

Hybrid functions divide the variables into groups and apply different
basic functions to each group.
"""

from typing import Any, Dict, List

import numpy as np

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
