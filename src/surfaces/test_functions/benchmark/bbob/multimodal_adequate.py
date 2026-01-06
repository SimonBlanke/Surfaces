# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""BBOB Multimodal Functions with Adequate Global Structure (f15-f19)."""

from typing import Any, Dict

import numpy as np

from ._base_bbob import BBOBFunction


class RastriginRotated(BBOBFunction):
    """f15: Rastrigin Function (Rotated).

    Rotated version of the Rastrigin function.
    The rotation breaks separability.

    Properties:
    - Highly multimodal (~10^D local optima)
    - Non-separable
    - Regular structure
    """

    _spec = {
        "name": "Rastrigin Function",
        "func_id": 15,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        Lambda = self.lambda_alpha(10)

        def rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self.R @ Lambda @ self.Q @ self.t_asy(self.t_osz(self.R @ (x - self.x_opt)), 0.2)
            D = self.n_dim
            return 10 * (D - np.sum(np.cos(2 * np.pi * z))) + np.sum(z**2) + self.f_opt

        self.pure_objective_function = rastrigin


class Weierstrass(BBOBFunction):
    """f16: Weierstrass Function.

    Continuous but nowhere differentiable function.
    Has self-similar structures at different scales.

    Properties:
    - Highly multimodal
    - Non-separable
    - Continuous but nowhere differentiable
    - Fractal structure
    """

    _spec = {
        "name": "Weierstrass Function",
        "func_id": 16,
        "unimodal": False,
        "separable": False,
        "differentiable": False,
    }

    def _create_objective_function(self) -> None:
        Lambda = self.lambda_alpha(100)
        k_max = 12
        a = 0.5
        b = 3

        # Precompute the offset
        f0 = sum(a**k * np.cos(np.pi * b**k) for k in range(k_max))

        def weierstrass(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self.R @ Lambda @ self.Q @ (x - self.x_opt)
            z = z * 0.01  # Scale to [-0.5, 0.5]

            D = self.n_dim
            result = 0.0
            for i in range(D):
                for k in range(k_max):
                    result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))

            result = 10 * ((result / D - f0) ** 3)
            return result + self.f_pen(x) / D + self.f_opt

        self.pure_objective_function = weierstrass


class SchaffersF7(BBOBFunction):
    """f17: Schaffer's F7 Function.

    Asymmetric multimodal function with irregular structure.
    Has global structure but many local optima.

    Properties:
    - Highly multimodal
    - Non-separable
    - Asymmetric
    """

    _spec = {
        "name": "Schaffer's F7 Function",
        "func_id": 17,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        Lambda = self.lambda_alpha(10)

        def schaffers_f7(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = Lambda @ self.Q @ self.t_asy(self.R @ (x - self.x_opt), 0.5)

            s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
            result = np.sum(np.sqrt(s) * (np.sin(50 * s**0.2) ** 2 + 1))
            result = (result / (self.n_dim - 1)) ** 2

            return result + self.f_pen(x) + self.f_opt

        self.pure_objective_function = schaffers_f7


class SchaffersF7Ill(BBOBFunction):
    """f18: Schaffer's F7 Function, Moderately Ill-Conditioned.

    Same as f17 but with higher condition number (1000).

    Properties:
    - Highly multimodal
    - Non-separable
    - Asymmetric
    - Ill-conditioned
    """

    _spec = {
        "name": "Schaffer's F7 Function, Ill-Conditioned",
        "func_id": 18,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        Lambda = self.lambda_alpha(1000)

        def schaffers_f7_ill(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = Lambda @ self.Q @ self.t_asy(self.R @ (x - self.x_opt), 0.5)

            s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
            result = np.sum(np.sqrt(s) * (np.sin(50 * s**0.2) ** 2 + 1))
            result = (result / (self.n_dim - 1)) ** 2

            return result + self.f_pen(x) + self.f_opt

        self.pure_objective_function = schaffers_f7_ill


class GriewankRosenbrock(BBOBFunction):
    """f19: Composite Griewank-Rosenbrock Function F8F2.

    Combines Griewank and Rosenbrock functions.
    Has funnel-like structure with irregular local optima.

    Properties:
    - Highly multimodal
    - Non-separable
    - Combines two functions
    """

    _spec = {
        "name": "Composite Griewank-Rosenbrock Function F8F2",
        "func_id": 19,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        c = max(1, np.sqrt(self.n_dim) / 8)

        def griewank_rosenbrock(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = c * self.R @ x + 0.5

            # Rosenbrock-like terms
            s = np.zeros(self.n_dim)
            for i in range(self.n_dim - 1):
                s[i] = 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
            # Wrap-around for last element
            s[-1] = 100 * (z[-1] ** 2 - z[0]) ** 2 + (z[-1] - 1) ** 2

            # Apply Griewank transformation
            result = np.sum(s / 4000 - np.cos(s)) + 1

            return 10 * result / self.n_dim + self.f_opt

        self.pure_objective_function = griewank_rosenbrock
