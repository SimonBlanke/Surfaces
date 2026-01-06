# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""BBOB Functions with Low or Moderate Conditioning (f6-f9)."""

from typing import Any, Dict

import numpy as np

from ._base_bbob import BBOBFunction


class AttractiveSector(BBOBFunction):
    """f6: Attractive Sector Function.

    Highly asymmetric function where the weights of the variables
    depend on their values.

    Properties:
    - Unimodal
    - Non-separable
    - Highly asymmetric
    """

    _spec = {
        "name": "Attractive Sector Function",
        "func_id": 6,
        "unimodal": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        Lambda = self.lambda_alpha(10)

        def attractive_sector(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self.Q @ Lambda @ self.R @ (x - self.x_opt)

            # Apply different weights based on sign
            s = np.where(z * self.x_opt > 0, 100, 1)
            result = np.sum((s * z) ** 2)
            return self.t_osz(result) ** 0.9 + self.f_opt

        self.pure_objective_function = attractive_sector


class StepEllipsoidal(BBOBFunction):
    """f7: Step Ellipsoidal Function.

    Ellipsoidal function with step-like structure.
    Has ~(2*5)^D plateaus.

    Properties:
    - Multi-modal (plateaus)
    - Non-separable
    - Ill-conditioned
    """

    _spec = {
        "name": "Step Ellipsoidal Function",
        "func_id": 7,
        "unimodal": False,
        "separable": False,
        "continuous": False,
    }

    def _create_objective_function(self) -> None:
        Lambda = self.lambda_alpha(10)

        def step_ellipsoidal(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z_hat = Lambda @ self.R @ (x - self.x_opt)

            # Apply step function
            z_tilde = np.where(
                np.abs(z_hat) > 0.5, np.floor(0.5 + z_hat), np.floor(0.5 + 10 * z_hat) / 10
            )

            z = self.Q @ z_tilde

            i = np.arange(self.n_dim)
            coeffs = np.power(10, 2 * i / (self.n_dim - 1)) if self.n_dim > 1 else np.ones(1)

            result = 0.1 * max(np.abs(z_hat[0]) / 1e4, np.sum(coeffs * z**2))
            return result + self.f_pen(x) + self.f_opt

        self.pure_objective_function = step_ellipsoidal


class RosenbrockOriginal(BBOBFunction):
    """f8: Rosenbrock Function, Original.

    The classic banana-shaped valley function.
    Non-convex with a narrow curved valley.

    Properties:
    - Unimodal in 2D, can have local optima in higher dimensions
    - Non-separable
    - Valley structure
    """

    _spec = {
        "name": "Rosenbrock Function, Original",
        "func_id": 8,
        "unimodal": False,
        "separable": False,
    }

    def _generate_x_opt(self) -> np.ndarray:
        """Generate x_opt constrained to [-3, 3]."""
        return self._rng.uniform(-3, 3, self.n_dim)

    def _create_objective_function(self) -> None:
        c = max(1, np.sqrt(self.n_dim) / 8)

        def rosenbrock(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = c * (x - self.x_opt) + 1

            result = 0.0
            for i in range(self.n_dim - 1):
                result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2

            return result + self.f_opt

        self.pure_objective_function = rosenbrock


class RosenbrockRotated(BBOBFunction):
    """f9: Rosenbrock Function, Rotated.

    Rotated version of the Rosenbrock function.
    The rotation makes the function non-separable in all variables.

    Properties:
    - Unimodal in 2D, can have local optima in higher dimensions
    - Non-separable
    - Valley structure (rotated)
    """

    _spec = {
        "name": "Rosenbrock Function, Rotated",
        "func_id": 9,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        c = max(1, np.sqrt(self.n_dim) / 8)

        def rosenbrock_rotated(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = c * self.R @ x + 0.5

            result = 0.0
            for i in range(self.n_dim - 1):
                result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2

            return result + self.f_opt

        self.pure_objective_function = rosenbrock_rotated
