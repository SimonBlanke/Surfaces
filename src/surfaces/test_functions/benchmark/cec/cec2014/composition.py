# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2014 Composition Functions (F23-F30).

Composition functions create complex landscapes by combining multiple
basic functions with different optima locations.
"""

from typing import Any, Dict, List

import numpy as np

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
            g1, g2, g3 = D // 3, D // 3, D - 2 * (D // 3)
            return (
                _high_conditioned_elliptic(z[:g1])
                + _bent_cigar(z[g1 : g1 + g2])
                + _rastrigin(z[g1 + g2 :])
            )

        def hybrid2(z: np.ndarray) -> float:
            # Similar to F18: Griewank + Weierstrass + Rosenbrock
            D = len(z)
            g1, g2, g3 = D // 3, D // 3, D - 2 * (D // 3)
            return _griewank(z[:g1]) + _weierstrass(z[g1 : g1 + g2]) + _rosenbrock(z[g1 + g2 :])

        def hybrid3(z: np.ndarray) -> float:
            # Similar to F19
            D = len(z)
            g1 = D // 4
            g2 = D // 4
            g3 = D // 4
            g4 = D - 3 * (D // 4)
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
            g4 = D - 3 * (D // 4)
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
