# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""BBOB Separable Functions (f1-f5)."""

from typing import Any, Dict

import numpy as np

from ._base_bbob import BBOBFunction


class Sphere(BBOBFunction):
    """f1: Sphere Function.

    The simplest unimodal function. Highly symmetric, smooth, and scalable.
    Used as a baseline for algorithm comparison.

    Properties:
    - Unimodal
    - Separable
    - Highly symmetric
    - Condition number: 1
    """

    _spec = {
        "name": "Sphere Function",
        "func_id": 1,
        "unimodal": True,
        "convex": True,
        "separable": True,
    }

    # Function sheet attributes
    latex_formula = r"f(\vec{x}) = \sum_{i=1}^{n} (x_i - x_i^*)^2 + f_{\text{opt}}"
    tagline = (
        "The BBOB baseline function. A simple shifted sphere used to "
        "calibrate algorithm performance on the easiest possible landscape."
    )
    display_bounds = (-5.0, 5.0)
    display_projection = {"fixed_value": 0.0}
    reference = "Hansen et al. (2009)"
    reference_url = "https://numbbo.github.io/coco/testsuites/bbob"

    def _create_objective_function(self) -> None:
        def sphere(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = x - self.x_opt
            return np.sum(z**2) + self.f_opt

        self.pure_objective_function = sphere


class EllipsoidalSeparable(BBOBFunction):
    """f2: Separable Ellipsoidal Function.

    Ill-conditioned separable function with condition number 10^6.
    Tests ability to exploit separability.

    Properties:
    - Unimodal
    - Separable
    - Ill-conditioned (condition number: 10^6)
    """

    _spec = {
        "name": "Separable Ellipsoidal Function",
        "func_id": 2,
        "unimodal": True,
        "convex": True,
        "separable": True,
    }

    def _create_objective_function(self) -> None:
        i = np.arange(self.n_dim)
        coeffs = np.power(1e6, i / (self.n_dim - 1)) if self.n_dim > 1 else np.ones(1)

        def ellipsoidal(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self.t_osz(x - self.x_opt)
            return np.sum(coeffs * z**2) + self.f_opt

        self.pure_objective_function = ellipsoidal


class RastriginSeparable(BBOBFunction):
    """f3: Rastrigin Function (Separable).

    Highly multimodal with regular local optima placement.
    Tests global search ability while being separable.

    Properties:
    - Highly multimodal (~10^D local optima)
    - Separable
    - Regular structure
    """

    _spec = {
        "name": "Rastrigin Function",
        "func_id": 3,
        "unimodal": False,
        "separable": True,
    }

    def _create_objective_function(self) -> None:
        Lambda = self.lambda_alpha(10)

        def rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = Lambda @ self.t_asy(self.t_osz(x - self.x_opt), 0.2)
            D = self.n_dim
            return 10 * (D - np.sum(np.cos(2 * np.pi * z))) + np.sum(z**2) + self.f_opt

        self.pure_objective_function = rastrigin


class BuecheRastrigin(BBOBFunction):
    """f4: Bueche-Rastrigin Function.

    Rastrigin variant with asymmetric transformation.
    Has ~10^D local optima.

    Properties:
    - Highly multimodal
    - Separable
    - Asymmetric
    """

    _spec = {
        "name": "Bueche-Rastrigin Function",
        "func_id": 4,
        "unimodal": False,
        "separable": True,
    }

    def _generate_x_opt(self) -> np.ndarray:
        """Generate x_opt with special structure for Bueche-Rastrigin."""
        x = self._rng.uniform(-4, 4, self.n_dim)
        # Make even-indexed components positive
        x[::2] = np.abs(x[::2])
        return x

    def _create_objective_function(self) -> None:
        Lambda = self.lambda_alpha(10)

        def bueche_rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = x - self.x_opt

            # Apply scaling to odd indices
            s = np.ones(self.n_dim)
            mask = (np.arange(self.n_dim) % 2 == 0) & (z > 0)
            s[mask] = 10

            z = s * self.t_osz(z)
            z = Lambda @ z

            D = self.n_dim
            result = 10 * (D - np.sum(np.cos(2 * np.pi * z))) + np.sum(z**2)
            return result + 100 * self.f_pen(x) + self.f_opt

        self.pure_objective_function = bueche_rastrigin


class LinearSlope(BBOBFunction):
    """f5: Linear Slope Function.

    The only non-quadratic purely convex function.
    Optimal point at the boundary of the domain.

    Properties:
    - Unimodal
    - Separable
    - Linear (only linear function in BBOB)
    - Optimum at boundary
    """

    _spec = {
        "name": "Linear Slope Function",
        "func_id": 5,
        "unimodal": True,
        "convex": True,
        "separable": True,
    }

    def _generate_x_opt(self) -> np.ndarray:
        """Generate x_opt at the boundary."""
        signs = np.where(self._rng.rand(self.n_dim) > 0.5, 1, -1)
        return 5 * signs

    def _create_objective_function(self) -> None:
        i = np.arange(self.n_dim)
        s = (
            np.sign(self.x_opt) * np.power(10, i / (self.n_dim - 1))
            if self.n_dim > 1
            else np.sign(self.x_opt)
        )

        def linear_slope(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            # Clip x to ensure we don't go past the boundary
            z = np.where(self.x_opt * x < 25, x, self.x_opt)
            return np.sum(5 * np.abs(s) - s * z) + self.f_opt

        self.pure_objective_function = linear_slope
