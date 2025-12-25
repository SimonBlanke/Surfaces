# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2014 Simple Multimodal Functions (F4-F16).

These functions have multiple local optima in addition to the global optimum.
"""

from typing import Any, Dict

import numpy as np

from ._base_cec2014 import CEC2014Function


class ShiftedRotatedRosenbrock(CEC2014Function):
    """F4: Shifted and Rotated Rosenbrock's Function.

    The Rosenbrock function has a narrow, parabolic valley. Finding the
    valley is trivial, but converging to the global optimum is difficult.

    Properties:
    - Multimodal (for D > 3)
    - Non-separable
    - Scalable
    - Has a narrow valley

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
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
            z = z * 2.048 / 100 + 1  # Scale to standard Rosenbrock domain

            result = 0.0
            for i in range(self.n_dim - 1):
                result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2

            return result + self.f_global

        self.pure_objective_function = rosenbrock


class ShiftedRotatedAckley(CEC2014Function):
    """F5: Shifted and Rotated Ackley's Function.

    The Ackley function has many local optima with a global optimum
    in a large basin of attraction.

    Properties:
    - Multimodal
    - Non-separable (due to rotation)
    - Scalable
    - Nearly flat outer region

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    """

    _spec = {
        "name": "Shifted and Rotated Ackley's Function",
        "func_id": 5,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def ackley(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            sum1 = np.sum(z**2)
            sum2 = np.sum(np.cos(2 * np.pi * z))
            D = self.n_dim

            result = -20 * np.exp(-0.2 * np.sqrt(sum1 / D)) - np.exp(sum2 / D) + 20 + np.e

            return result + self.f_global

        self.pure_objective_function = ackley


class ShiftedRotatedWeierstrass(CEC2014Function):
    """F6: Shifted and Rotated Weierstrass Function.

    A continuous but nowhere differentiable function with a
    fractal-like structure.

    Properties:
    - Multimodal
    - Non-separable (due to rotation)
    - Scalable
    - Continuous but not differentiable

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    """

    _spec = {
        "name": "Shifted and Rotated Weierstrass Function",
        "func_id": 6,
        "unimodal": False,
        "separable": False,
        "differentiable": False,
    }

    def _create_objective_function(self) -> None:
        # Precompute constants
        a = 0.5
        b = 3
        k_max = 20

        def weierstrass(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 0.5 / 100  # Scale

            result = 0.0
            for i in range(self.n_dim):
                for k in range(k_max + 1):
                    result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))

            # Subtract the offset
            offset = 0.0
            for k in range(k_max + 1):
                offset += a**k * np.cos(2 * np.pi * b**k * 0.5)
            result -= self.n_dim * offset

            return result + self.f_global

        self.pure_objective_function = weierstrass


class ShiftedRotatedGriewank(CEC2014Function):
    """F7: Shifted and Rotated Griewank's Function.

    The Griewank function has many widespread local optima regularly
    distributed.

    Properties:
    - Multimodal
    - Non-separable (due to rotation)
    - Scalable

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    """

    _spec = {
        "name": "Shifted and Rotated Griewank's Function",
        "func_id": 7,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def griewank(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 600 / 100  # Scale to standard Griewank domain

            sum_sq = np.sum(z**2) / 4000
            prod_cos = np.prod(np.cos(z / np.sqrt(np.arange(1, self.n_dim + 1))))

            return sum_sq - prod_cos + 1 + self.f_global

        self.pure_objective_function = griewank


class ShiftedRastrigin(CEC2014Function):
    """F8: Shifted Rastrigin's Function.

    The Rastrigin function is highly multimodal with local optima
    arranged in a regular lattice pattern.

    Properties:
    - Highly multimodal
    - Separable (no rotation applied)
    - Scalable

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    """

    _spec = {
        "name": "Shifted Rastrigin's Function",
        "func_id": 8,
        "unimodal": False,
        "separable": True,  # No rotation
    }

    def _create_objective_function(self) -> None:
        def rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x)  # Only shift, no rotation
            z = z * 5.12 / 100  # Scale

            result = 10 * self.n_dim + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

            return result + self.f_global

        self.pure_objective_function = rastrigin


class ShiftedRotatedRastrigin(CEC2014Function):
    """F9: Shifted and Rotated Rastrigin's Function.

    A rotated version of the Rastrigin function, making it non-separable.

    Properties:
    - Highly multimodal
    - Non-separable (due to rotation)
    - Scalable

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    """

    _spec = {
        "name": "Shifted and Rotated Rastrigin's Function",
        "func_id": 9,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 5.12 / 100  # Scale

            result = 10 * self.n_dim + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

            return result + self.f_global

        self.pure_objective_function = rastrigin


class ShiftedSchwefel(CEC2014Function):
    """F10: Shifted Schwefel's Function.

    The Schwefel function has a second-best optimum far from the
    global optimum, which can trap algorithms.

    Properties:
    - Multimodal
    - Separable (no rotation)
    - Scalable
    - Deceptive (has misleading local optima)

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    """

    _spec = {
        "name": "Shifted Schwefel's Function",
        "func_id": 10,
        "unimodal": False,
        "separable": True,
    }

    def _create_objective_function(self) -> None:
        def schwefel(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x)  # Only shift
            z = z * 1000 / 100 + 4.209687462275036e2  # Scale and shift

            result = 0.0
            for i in range(self.n_dim):
                zi = z[i]
                if abs(zi) <= 500:
                    result += zi * np.sin(np.sqrt(abs(zi)))
                elif zi > 500:
                    result += (500 - zi % 500) * np.sin(np.sqrt(abs(500 - zi % 500))) - (
                        zi - 500
                    ) ** 2 / (10000 * self.n_dim)
                else:
                    result += (abs(zi) % 500 - 500) * np.sin(np.sqrt(abs(abs(zi) % 500 - 500))) - (
                        zi + 500
                    ) ** 2 / (10000 * self.n_dim)

            result = 418.9829 * self.n_dim - result

            return result + self.f_global

        self.pure_objective_function = schwefel


class ShiftedRotatedSchwefel(CEC2014Function):
    """F11: Shifted and Rotated Schwefel's Function.

    A rotated version of the Schwefel function.

    Properties:
    - Multimodal
    - Non-separable (due to rotation)
    - Scalable
    - Deceptive

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    """

    _spec = {
        "name": "Shifted and Rotated Schwefel's Function",
        "func_id": 11,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def schwefel(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 1000 / 100 + 4.209687462275036e2

            result = 0.0
            for i in range(self.n_dim):
                zi = z[i]
                if abs(zi) <= 500:
                    result += zi * np.sin(np.sqrt(abs(zi)))
                elif zi > 500:
                    result += (500 - zi % 500) * np.sin(np.sqrt(abs(500 - zi % 500))) - (
                        zi - 500
                    ) ** 2 / (10000 * self.n_dim)
                else:
                    result += (abs(zi) % 500 - 500) * np.sin(np.sqrt(abs(abs(zi) % 500 - 500))) - (
                        zi + 500
                    ) ** 2 / (10000 * self.n_dim)

            result = 418.9829 * self.n_dim - result

            return result + self.f_global

        self.pure_objective_function = schwefel


class ShiftedRotatedKatsuura(CEC2014Function):
    """F12: Shifted and Rotated Katsuura Function.

    A continuous but nowhere differentiable function.

    Properties:
    - Multimodal
    - Non-separable (due to rotation)
    - Scalable
    - Continuous but not differentiable

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    """

    _spec = {
        "name": "Shifted and Rotated Katsuura Function",
        "func_id": 12,
        "unimodal": False,
        "separable": False,
        "differentiable": False,
    }

    def _create_objective_function(self) -> None:
        def katsuura(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 5 / 100  # Scale

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


class ShiftedRotatedHappyCat(CEC2014Function):
    """F13: Shifted and Rotated HappyCat Function.

    A function that resembles a smiling cat when plotted in 2D.

    Properties:
    - Multimodal
    - Non-separable (due to rotation)
    - Scalable

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    """

    _spec = {
        "name": "Shifted and Rotated HappyCat Function",
        "func_id": 13,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        alpha = 1.0 / 8.0

        def happycat(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 5 / 100 - 1  # Scale and shift to move optimum to origin

            D = self.n_dim
            sum_sq = np.sum(z**2)
            sum_z = np.sum(z)

            result = abs(sum_sq - D) ** (2 * alpha) + (0.5 * sum_sq + sum_z) / D + 0.5

            return result + self.f_global

        self.pure_objective_function = happycat


class ShiftedRotatedHGBat(CEC2014Function):
    """F14: Shifted and Rotated HGBat Function.

    A function with a bat-like shape when plotted.

    Properties:
    - Multimodal
    - Non-separable (due to rotation)
    - Scalable

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    """

    _spec = {
        "name": "Shifted and Rotated HGBat Function",
        "func_id": 14,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def hgbat(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 5 / 100 - 1  # Scale and shift to move optimum to origin

            D = self.n_dim
            sum_sq = np.sum(z**2)
            sum_z = np.sum(z)

            result = abs(sum_sq**2 - sum_z**2) ** 0.5 + (0.5 * sum_sq + sum_z) / D + 0.5

            return result + self.f_global

        self.pure_objective_function = hgbat


class ShiftedRotatedExpandedGriewankRosenbrock(CEC2014Function):
    """F15: Shifted and Rotated Expanded Griewank's plus Rosenbrock's Function.

    A composition of Griewank and Rosenbrock functions.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    """

    _spec = {
        "name": "Shifted and Rotated Expanded Griewank's plus Rosenbrock's Function",
        "func_id": 15,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def griewank_rosenbrock(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = z * 5 / 100 + 1  # Scale

            result = 0.0
            for i in range(self.n_dim - 1):
                # Rosenbrock term
                t = 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
                # Griewank of Rosenbrock
                result += t**2 / 4000 - np.cos(t) + 1

            # Last term wraps around
            t = 100 * (z[-1] ** 2 - z[0]) ** 2 + (z[-1] - 1) ** 2
            result += t**2 / 4000 - np.cos(t) + 1

            return result + self.f_global

        self.pure_objective_function = griewank_rosenbrock


class ShiftedRotatedExpandedScafferF6(CEC2014Function):
    """F16: Shifted and Rotated Expanded Scaffer's F6 Function.

    An expanded version of the Schaffer F6 function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    """

    _spec = {
        "name": "Shifted and Rotated Expanded Scaffer's F6 Function",
        "func_id": 16,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def schaffer_f6(x1: float, x2: float) -> float:
            t = x1**2 + x2**2
            return 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2

        def expanded_schaffer(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            result = 0.0
            for i in range(self.n_dim - 1):
                result += schaffer_f6(z[i], z[i + 1])
            result += schaffer_f6(z[-1], z[0])

            return result + self.f_global

        self.pure_objective_function = expanded_schaffer
