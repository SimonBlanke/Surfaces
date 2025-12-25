# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2013 Benchmark Functions (F1-F28)."""

from typing import Any, Dict

import numpy as np

from ._base_cec2013 import CEC2013Function

# =============================================================================
# Unimodal Functions (F1-F5)
# =============================================================================


class Sphere(CEC2013Function):
    """F1: Sphere Function.

    Properties:
    - Unimodal
    - Separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = False

    _spec = {
        "name": "Sphere Function",
        "func_id": 1,
        "unimodal": True,
        "convex": True,
        "separable": True,
    }

    def _create_objective_function(self) -> None:
        def sphere(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x)
            return np.sum(z**2) + self.f_global

        self.pure_objective_function = sphere


class RotatedHighConditionedElliptic(CEC2013Function):
    """F2: Rotated High Conditioned Elliptic Function.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated High Conditioned Elliptic Function",
        "func_id": 2,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def elliptic(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._oscillation(self._shift_rotate(x))

            D = self.n_dim
            result = 0.0
            for i in range(D):
                result += (10**6) ** (i / (D - 1)) * z[i] ** 2

            return result + self.f_global

        self.pure_objective_function = elliptic


class RotatedBentCigar(CEC2013Function):
    """F3: Rotated Bent Cigar Function.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Bent Cigar Function",
        "func_id": 3,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def bent_cigar(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._asymmetric(self._shift_rotate(x), 0.5)
            z = self._rotate(z)

            return z[0] ** 2 + 10**6 * np.sum(z[1:] ** 2) + self.f_global

        self.pure_objective_function = bent_cigar


class RotatedDiscus(CEC2013Function):
    """F4: Rotated Discus Function.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Discus Function",
        "func_id": 4,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def discus(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._oscillation(self._shift_rotate(x))

            return 10**6 * z[0] ** 2 + np.sum(z[1:] ** 2) + self.f_global

        self.pure_objective_function = discus


class DifferentPowers(CEC2013Function):
    """F5: Different Powers Function.

    Properties:
    - Unimodal
    - Separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = False

    _spec = {
        "name": "Different Powers Function",
        "func_id": 5,
        "unimodal": True,
        "separable": True,
    }

    def _create_objective_function(self) -> None:
        def different_powers(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x)

            D = self.n_dim
            result = 0.0
            for i in range(D):
                result += abs(z[i]) ** (2 + 4 * i / (D - 1))

            return np.sqrt(result) + self.f_global

        self.pure_objective_function = different_powers


# =============================================================================
# Multimodal Functions (F6-F20)
# =============================================================================


class RotatedRosenbrock(CEC2013Function):
    """F6: Rotated Rosenbrock's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Rosenbrock's Function",
        "func_id": 6,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def rosenbrock(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 2.048 / 100 + 1

            result = 0.0
            for i in range(self.n_dim - 1):
                result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2

            return result + self.f_global

        self.pure_objective_function = rosenbrock


class RotatedSchafferF7(CEC2013Function):
    """F7: Rotated Schaffer's F7 Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Schaffer's F7 Function",
        "func_id": 7,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def schaffer_f7(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._asymmetric(self._shift_rotate(x), 0.5)
            z = self._rotate(z)

            D = self.n_dim
            s = np.zeros(D - 1)
            for i in range(D - 1):
                s[i] = np.sqrt(z[i] ** 2 + z[i + 1] ** 2)

            result = 0.0
            for i in range(D - 1):
                result += np.sqrt(s[i]) * (np.sin(50 * s[i] ** 0.2) + 1)

            result = (result / (D - 1)) ** 2

            return result + self.f_global

        self.pure_objective_function = schaffer_f7


class RotatedAckley(CEC2013Function):
    """F8: Rotated Ackley's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Ackley's Function",
        "func_id": 8,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def ackley(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._asymmetric(self._shift_rotate(x), 0.5)
            z = self._rotate(z)

            D = self.n_dim
            sum1 = np.sum(z**2)
            sum2 = np.sum(np.cos(2 * np.pi * z))

            result = -20 * np.exp(-0.2 * np.sqrt(sum1 / D)) - np.exp(sum2 / D) + 20 + np.e

            return result + self.f_global

        self.pure_objective_function = ackley


class RotatedWeierstrass(CEC2013Function):
    """F9: Rotated Weierstrass Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    - Continuous but not differentiable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Weierstrass Function",
        "func_id": 9,
        "unimodal": False,
        "separable": False,
        "differentiable": False,
    }

    def _create_objective_function(self) -> None:
        a = 0.5
        b = 3
        k_max = 20

        def weierstrass(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._asymmetric(self._shift_rotate(x), 0.5)
            z = self._rotate(z)
            z = z * 0.5 / 100

            D = self.n_dim
            result = 0.0
            for i in range(D):
                for k in range(k_max + 1):
                    result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))

            offset = D * sum(a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1))
            result -= offset

            return result + self.f_global

        self.pure_objective_function = weierstrass


class RotatedGriewank(CEC2013Function):
    """F10: Rotated Griewank's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Griewank's Function",
        "func_id": 10,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def griewank(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 600 / 100

            D = self.n_dim
            sum_sq = np.sum(z**2) / 4000
            prod_cos = np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1))))

            return sum_sq - prod_cos + 1 + self.f_global

        self.pure_objective_function = griewank


class Rastrigin(CEC2013Function):
    """F11: Rastrigin's Function (non-rotated).

    Properties:
    - Highly multimodal
    - Separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = False

    _spec = {
        "name": "Rastrigin's Function",
        "func_id": 11,
        "unimodal": False,
        "separable": True,
    }

    def _create_objective_function(self) -> None:
        def rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._oscillation(self._asymmetric(self._shift(x), 0.2))
            z = z * 5.12 / 100

            D = self.n_dim
            result = 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

            return result + self.f_global

        self.pure_objective_function = rastrigin


class RotatedRastrigin(CEC2013Function):
    """F12: Rotated Rastrigin's Function.

    Properties:
    - Highly multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Rastrigin's Function",
        "func_id": 12,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._oscillation(self._asymmetric(self._shift_rotate(x), 0.2))
            z = self._rotate(z)
            z = z * 5.12 / 100

            D = self.n_dim
            result = 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

            return result + self.f_global

        self.pure_objective_function = rastrigin


class StepRastrigin(CEC2013Function):
    """F13: Non-Continuous Rotated Rastrigin's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Non-continuous
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Non-Continuous Rotated Rastrigin's Function",
        "func_id": 13,
        "unimodal": False,
        "separable": False,
        "continuous": False,
    }

    def _create_objective_function(self) -> None:
        def step_rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._oscillation(self._asymmetric(self._shift_rotate(x), 0.2))
            z = self._rotate(z)
            z = z * 5.12 / 100

            # Step function transformation
            y = z.copy()
            for i in range(len(z)):
                if abs(y[i]) > 0.5:
                    y[i] = np.round(2 * y[i]) / 2

            D = self.n_dim
            result = 10 * D + np.sum(y**2 - 10 * np.cos(2 * np.pi * y))

            return result + self.f_global

        self.pure_objective_function = step_rastrigin


class Schwefel(CEC2013Function):
    """F14: Schwefel's Function (non-rotated).

    Properties:
    - Multimodal
    - Separable
    - Deceptive
    """

    shift_index = 1
    uses_rotation = False

    _spec = {
        "name": "Schwefel's Function",
        "func_id": 14,
        "unimodal": False,
        "separable": True,
    }

    def _create_objective_function(self) -> None:
        def schwefel(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x)
            z = z * 1000 / 100 + 4.209687462275036e2

            D = self.n_dim
            result = 0.0
            for i in range(D):
                zi = z[i]
                if abs(zi) <= 500:
                    result += zi * np.sin(np.sqrt(abs(zi)))
                elif zi > 500:
                    result += (500 - zi % 500) * np.sin(np.sqrt(abs(500 - zi % 500)))
                    result -= (zi - 500) ** 2 / (10000 * D)
                else:
                    result += (abs(zi) % 500 - 500) * np.sin(np.sqrt(abs(abs(zi) % 500 - 500)))
                    result -= (zi + 500) ** 2 / (10000 * D)

            result = 418.9829 * D - result

            return result + self.f_global

        self.pure_objective_function = schwefel


class RotatedSchwefel(CEC2013Function):
    """F15: Rotated Schwefel's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Deceptive
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Schwefel's Function",
        "func_id": 15,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def schwefel(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 1000 / 100 + 4.209687462275036e2

            D = self.n_dim
            result = 0.0
            for i in range(D):
                zi = z[i]
                if abs(zi) <= 500:
                    result += zi * np.sin(np.sqrt(abs(zi)))
                elif zi > 500:
                    result += (500 - zi % 500) * np.sin(np.sqrt(abs(500 - zi % 500)))
                    result -= (zi - 500) ** 2 / (10000 * D)
                else:
                    result += (abs(zi) % 500 - 500) * np.sin(np.sqrt(abs(abs(zi) % 500 - 500)))
                    result -= (zi + 500) ** 2 / (10000 * D)

            result = 418.9829 * D - result

            return result + self.f_global

        self.pure_objective_function = schwefel


class RotatedKatsuura(CEC2013Function):
    """F16: Rotated Katsuura Function.

    Properties:
    - Multimodal
    - Non-separable
    - Non-differentiable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Katsuura Function",
        "func_id": 16,
        "unimodal": False,
        "separable": False,
        "differentiable": False,
    }

    def _create_objective_function(self) -> None:
        def katsuura(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 5 / 100

            D = self.n_dim
            result = 1.0
            for i in range(D):
                inner_sum = 0.0
                for j in range(1, 33):
                    inner_sum += abs(2**j * z[i] - round(2**j * z[i])) / (2**j)
                result *= (1 + (i + 1) * inner_sum) ** (10 / (D**1.2))

            result = (10 / D**2) * result - (10 / D**2)

            return result + self.f_global

        self.pure_objective_function = katsuura


class LunacekBiRastrigin(CEC2013Function):
    """F17: Lunacek Bi-Rastrigin Function (non-rotated).

    Properties:
    - Multimodal
    - Separable
    - Two global optima
    """

    shift_index = 1
    uses_rotation = False

    _spec = {
        "name": "Lunacek Bi-Rastrigin Function",
        "func_id": 17,
        "unimodal": False,
        "separable": True,
    }

    def _create_objective_function(self) -> None:
        def bi_rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x)
            z = z * 10 / 100

            D = self.n_dim
            mu0 = 2.5
            s = 1 - 1 / (2 * np.sqrt(D + 20) - 8.2)
            mu1 = -np.sqrt((mu0**2 - 1) / s)
            d = 1

            sum1 = np.sum((z - mu0) ** 2)
            sum2 = np.sum((z - mu1) ** 2)
            sum3 = np.sum(np.cos(2 * np.pi * (z - mu0)))

            result = min(sum1, d * D + s * sum2) + 10 * (D - sum3)

            return result + self.f_global

        self.pure_objective_function = bi_rastrigin


class RotatedLunacekBiRastrigin(CEC2013Function):
    """F18: Rotated Lunacek Bi-Rastrigin Function.

    Properties:
    - Multimodal
    - Non-separable
    - Two global optima
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Lunacek Bi-Rastrigin Function",
        "func_id": 18,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def bi_rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 10 / 100

            D = self.n_dim
            mu0 = 2.5
            s = 1 - 1 / (2 * np.sqrt(D + 20) - 8.2)
            mu1 = -np.sqrt((mu0**2 - 1) / s)
            d = 1

            y = self._rotate(z - mu0)

            sum1 = np.sum((z - mu0) ** 2)
            sum2 = np.sum((z - mu1) ** 2)
            sum3 = np.sum(np.cos(2 * np.pi * y))

            result = min(sum1, d * D + s * sum2) + 10 * (D - sum3)

            return result + self.f_global

        self.pure_objective_function = bi_rastrigin


class RotatedExpandedGriewankRosenbrock(CEC2013Function):
    """F19: Rotated Expanded Griewank's plus Rosenbrock's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Expanded Griewank's plus Rosenbrock's Function",
        "func_id": 19,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def griewank_rosenbrock(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 5 / 100 + 1

            D = self.n_dim
            result = 0.0
            for i in range(D - 1):
                t = 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
                result += t**2 / 4000 - np.cos(t) + 1

            t = 100 * (z[-1] ** 2 - z[0]) ** 2 + (z[-1] - 1) ** 2
            result += t**2 / 4000 - np.cos(t) + 1

            return result + self.f_global

        self.pure_objective_function = griewank_rosenbrock


class RotatedExpandedScafferF6(CEC2013Function):
    """F20: Rotated Expanded Scaffer's F6 Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "name": "Rotated Expanded Scaffer's F6 Function",
        "func_id": 20,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def schaffer_f6(x1: float, x2: float) -> float:
            t = x1**2 + x2**2
            return 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2

        def expanded_scaffer(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._asymmetric(self._shift_rotate(x), 0.5)
            z = self._rotate(z)

            D = self.n_dim
            result = 0.0
            for i in range(D - 1):
                result += schaffer_f6(z[i], z[i + 1])
            result += schaffer_f6(z[-1], z[0])

            return result + self.f_global

        self.pure_objective_function = expanded_scaffer


# =============================================================================
# Composition Functions (F21-F28)
# =============================================================================


class _CompositionBase(CEC2013Function):
    """Base class for CEC 2013 composition functions."""

    _spec = {
        "unimodal": False,
        "separable": False,
    }

    n_functions: int = 0
    sigmas: list = []
    lambdas: list = []
    biases: list = []

    def _compute_weights(self, x: np.ndarray) -> np.ndarray:
        """Compute weights for each component function."""
        weights = np.zeros(self.n_functions)
        data = self._load_data()

        for i in range(self.n_functions):
            shift = data.get(f"shift_{i + 1}", np.zeros(self.n_dim))
            diff = x - shift
            dist_sq = np.sum(diff**2)
            if dist_sq != 0:
                weights[i] = np.exp(-dist_sq / (2 * self.n_dim * self.sigmas[i] ** 2))
            else:
                weights[i] = 1e10

        max_weight = np.max(weights)
        if max_weight == 0:
            weights = np.ones(self.n_functions) / self.n_functions
        else:
            for i in range(self.n_functions):
                if weights[i] != max_weight:
                    weights[i] *= 1 - max_weight**10
            weights = weights / np.sum(weights)

        return weights


class CompositionFunction1(_CompositionBase):
    """F21: Composition Function 1.

    Combines: Rosenbrock, High Conditioned Elliptic, Bent Cigar,
    Discus, High Conditioned Elliptic.
    """

    _spec = {
        "name": "Composition Function 1",
        "func_id": 21,
    }

    n_functions = 5
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [1, 1e-6, 1e-26, 1e-6, 1e-6]
    biases = [0, 100, 200, 300, 400]

    def _create_objective_function(self) -> None:
        def rosenbrock(z: np.ndarray) -> float:
            z = z * 2.048 / 100 + 1
            return sum(
                100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2 for i in range(len(z) - 1)
            )

        def elliptic(z: np.ndarray) -> float:
            D = len(z)
            return sum((10**6) ** (i / (D - 1)) * z[i] ** 2 for i in range(D))

        def bent_cigar(z: np.ndarray) -> float:
            return z[0] ** 2 + 10**6 * np.sum(z[1:] ** 2)

        def discus(z: np.ndarray) -> float:
            return 10**6 * z[0] ** 2 + np.sum(z[1:] ** 2)

        functions = [rosenbrock, elliptic, bent_cigar, discus, elliptic]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            weights = self._compute_weights(x)
            data = self._load_data()

            result = 0.0
            for i in range(self.n_functions):
                shift = data.get(f"shift_{i + 1}", np.zeros(self.n_dim))
                M = data.get(f"rotation_{i + 1}", np.eye(self.n_dim))
                z = M @ (x - shift)
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition


class CompositionFunction2(_CompositionBase):
    """F22: Composition Function 2.

    Combines: Schwefel, Rastrigin, High Conditioned Elliptic.
    """

    _spec = {
        "name": "Composition Function 2",
        "func_id": 22,
    }

    n_functions = 3
    sigmas = [20, 20, 20]
    lambdas = [10, 1, 1e-6]
    biases = [0, 100, 200]

    def _create_objective_function(self) -> None:
        def schwefel(z: np.ndarray) -> float:
            z = z * 1000 / 100 + 4.209687462275036e2
            D = len(z)
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

        def rastrigin(z: np.ndarray) -> float:
            z = z * 5.12 / 100
            D = len(z)
            return 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

        def elliptic(z: np.ndarray) -> float:
            D = len(z)
            return sum((10**6) ** (i / (D - 1)) * z[i] ** 2 for i in range(D))

        functions = [schwefel, rastrigin, elliptic]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            weights = self._compute_weights(x)
            data = self._load_data()

            result = 0.0
            for i in range(self.n_functions):
                shift = data.get(f"shift_{i + 1}", np.zeros(self.n_dim))
                z = x - shift  # No rotation for F22
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition


class CompositionFunction3(_CompositionBase):
    """F23: Composition Function 3."""

    _spec = {
        "name": "Composition Function 3",
        "func_id": 23,
    }

    n_functions = 3
    sigmas = [20, 20, 20]
    lambdas = [10, 1, 1e-6]
    biases = [0, 100, 200]

    def _create_objective_function(self) -> None:
        def schwefel(z: np.ndarray) -> float:
            z = z * 1000 / 100 + 4.209687462275036e2
            D = len(z)
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

        def rastrigin(z: np.ndarray) -> float:
            z = z * 5.12 / 100
            D = len(z)
            return 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

        def elliptic(z: np.ndarray) -> float:
            D = len(z)
            return sum((10**6) ** (i / (D - 1)) * z[i] ** 2 for i in range(D))

        functions = [schwefel, rastrigin, elliptic]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            weights = self._compute_weights(x)
            data = self._load_data()

            result = 0.0
            for i in range(self.n_functions):
                shift = data.get(f"shift_{i + 1}", np.zeros(self.n_dim))
                M = data.get(f"rotation_{i + 1}", np.eye(self.n_dim))
                z = M @ (x - shift)
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition


class CompositionFunction4(_CompositionBase):
    """F24: Composition Function 4."""

    _spec = {
        "name": "Composition Function 4",
        "func_id": 24,
    }

    n_functions = 3
    sigmas = [20, 20, 20]
    lambdas = [10, 1, 1e-6]
    biases = [0, 100, 200]

    def _create_objective_function(self) -> None:
        def schwefel(z: np.ndarray) -> float:
            z = z * 1000 / 100 + 4.209687462275036e2
            D = len(z)
            result = 0.0
            for zi in z:
                if abs(zi) <= 500:
                    result += zi * np.sin(np.sqrt(abs(zi)))
            return 418.9829 * D - result

        def rastrigin(z: np.ndarray) -> float:
            z = z * 5.12 / 100
            D = len(z)
            return 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

        def weierstrass(z: np.ndarray) -> float:
            a, b, k_max = 0.5, 3, 20
            D = len(z)
            z = z * 0.5 / 100
            result = 0.0
            for i in range(D):
                for k in range(k_max + 1):
                    result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))
            offset = D * sum(a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1))
            return result - offset

        functions = [schwefel, rastrigin, weierstrass]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            weights = self._compute_weights(x)
            data = self._load_data()

            result = 0.0
            for i in range(self.n_functions):
                shift = data.get(f"shift_{i + 1}", np.zeros(self.n_dim))
                M = data.get(f"rotation_{i + 1}", np.eye(self.n_dim))
                z = M @ (x - shift)
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition


class CompositionFunction5(_CompositionBase):
    """F25: Composition Function 5."""

    _spec = {
        "name": "Composition Function 5",
        "func_id": 25,
    }

    n_functions = 3
    sigmas = [10, 30, 50]
    lambdas = [10, 1, 1]
    biases = [0, 100, 200]

    def _create_objective_function(self) -> None:
        def schwefel(z: np.ndarray) -> float:
            z = z * 1000 / 100 + 4.209687462275036e2
            D = len(z)
            result = 0.0
            for zi in z:
                if abs(zi) <= 500:
                    result += zi * np.sin(np.sqrt(abs(zi)))
            return 418.9829 * D - result

        def rastrigin(z: np.ndarray) -> float:
            z = z * 5.12 / 100
            D = len(z)
            return 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

        def ackley(z: np.ndarray) -> float:
            D = len(z)
            sum1 = np.sum(z**2)
            sum2 = np.sum(np.cos(2 * np.pi * z))
            return -20 * np.exp(-0.2 * np.sqrt(sum1 / D)) - np.exp(sum2 / D) + 20 + np.e

        functions = [schwefel, rastrigin, ackley]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            weights = self._compute_weights(x)
            data = self._load_data()

            result = 0.0
            for i in range(self.n_functions):
                shift = data.get(f"shift_{i + 1}", np.zeros(self.n_dim))
                M = data.get(f"rotation_{i + 1}", np.eye(self.n_dim))
                z = M @ (x - shift)
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition


class CompositionFunction6(_CompositionBase):
    """F26: Composition Function 6."""

    _spec = {
        "name": "Composition Function 6",
        "func_id": 26,
    }

    n_functions = 5
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [10, 1, 1e-6, 1, 1]
    biases = [0, 100, 200, 300, 400]

    def _create_objective_function(self) -> None:
        def schwefel(z: np.ndarray) -> float:
            z = z * 1000 / 100 + 4.209687462275036e2
            D = len(z)
            result = 0.0
            for zi in z:
                if abs(zi) <= 500:
                    result += zi * np.sin(np.sqrt(abs(zi)))
            return 418.9829 * D - result

        def rastrigin(z: np.ndarray) -> float:
            z = z * 5.12 / 100
            D = len(z)
            return 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

        def elliptic(z: np.ndarray) -> float:
            D = len(z)
            return sum((10**6) ** (i / (D - 1)) * z[i] ** 2 for i in range(D))

        def weierstrass(z: np.ndarray) -> float:
            a, b, k_max = 0.5, 3, 20
            D = len(z)
            z = z * 0.5 / 100
            result = 0.0
            for i in range(D):
                for k in range(k_max + 1):
                    result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))
            offset = D * sum(a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1))
            return result - offset

        def griewank(z: np.ndarray) -> float:
            D = len(z)
            z = z * 600 / 100
            return np.sum(z**2) / 4000 - np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1)))) + 1

        functions = [schwefel, rastrigin, elliptic, weierstrass, griewank]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            weights = self._compute_weights(x)
            data = self._load_data()

            result = 0.0
            for i in range(self.n_functions):
                shift = data.get(f"shift_{i + 1}", np.zeros(self.n_dim))
                M = data.get(f"rotation_{i + 1}", np.eye(self.n_dim))
                z = M @ (x - shift)
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition


class CompositionFunction7(_CompositionBase):
    """F27: Composition Function 7."""

    _spec = {
        "name": "Composition Function 7",
        "func_id": 27,
    }

    n_functions = 5
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [10, 10, 2.5, 25, 1e-6]
    biases = [0, 100, 200, 300, 400]

    def _create_objective_function(self) -> None:
        def griewank(z: np.ndarray) -> float:
            D = len(z)
            z = z * 600 / 100
            return np.sum(z**2) / 4000 - np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1)))) + 1

        def rastrigin(z: np.ndarray) -> float:
            z = z * 5.12 / 100
            D = len(z)
            return 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

        def schwefel(z: np.ndarray) -> float:
            z = z * 1000 / 100 + 4.209687462275036e2
            D = len(z)
            result = 0.0
            for zi in z:
                if abs(zi) <= 500:
                    result += zi * np.sin(np.sqrt(abs(zi)))
            return 418.9829 * D - result

        def weierstrass(z: np.ndarray) -> float:
            a, b, k_max = 0.5, 3, 20
            D = len(z)
            z = z * 0.5 / 100
            result = 0.0
            for i in range(D):
                for k in range(k_max + 1):
                    result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))
            offset = D * sum(a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1))
            return result - offset

        def elliptic(z: np.ndarray) -> float:
            D = len(z)
            return sum((10**6) ** (i / (D - 1)) * z[i] ** 2 for i in range(D))

        functions = [griewank, rastrigin, schwefel, weierstrass, elliptic]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            weights = self._compute_weights(x)
            data = self._load_data()

            result = 0.0
            for i in range(self.n_functions):
                shift = data.get(f"shift_{i + 1}", np.zeros(self.n_dim))
                M = data.get(f"rotation_{i + 1}", np.eye(self.n_dim))
                z = M @ (x - shift)
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition


class CompositionFunction8(_CompositionBase):
    """F28: Composition Function 8."""

    _spec = {
        "name": "Composition Function 8",
        "func_id": 28,
    }

    n_functions = 5
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [10, 10, 2.5, 25, 1e-6]
    biases = [0, 100, 200, 300, 400]

    def _create_objective_function(self) -> None:
        def ackley(z: np.ndarray) -> float:
            D = len(z)
            sum1 = np.sum(z**2)
            sum2 = np.sum(np.cos(2 * np.pi * z))
            return -20 * np.exp(-0.2 * np.sqrt(sum1 / D)) - np.exp(sum2 / D) + 20 + np.e

        def griewank(z: np.ndarray) -> float:
            D = len(z)
            z = z * 600 / 100
            return np.sum(z**2) / 4000 - np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1)))) + 1

        def schwefel(z: np.ndarray) -> float:
            z = z * 1000 / 100 + 4.209687462275036e2
            D = len(z)
            result = 0.0
            for zi in z:
                if abs(zi) <= 500:
                    result += zi * np.sin(np.sqrt(abs(zi)))
            return 418.9829 * D - result

        def rastrigin(z: np.ndarray) -> float:
            z = z * 5.12 / 100
            D = len(z)
            return 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

        def elliptic(z: np.ndarray) -> float:
            D = len(z)
            return sum((10**6) ** (i / (D - 1)) * z[i] ** 2 for i in range(D))

        functions = [ackley, griewank, schwefel, rastrigin, elliptic]

        def composition(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            weights = self._compute_weights(x)
            data = self._load_data()

            result = 0.0
            for i in range(self.n_functions):
                shift = data.get(f"shift_{i + 1}", np.zeros(self.n_dim))
                M = data.get(f"rotation_{i + 1}", np.eye(self.n_dim))
                z = M @ (x - shift)
                f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
                result += weights[i] * f_val

            return result + self.f_global

        self.pure_objective_function = composition

    name = "Composition Function 1"
    func_id = 21
