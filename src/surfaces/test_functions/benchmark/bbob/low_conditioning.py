# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""BBOB Functions with Low or Moderate Conditioning (f6-f9)."""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from .._batch_transforms import batch_f_pen, batch_lambda_alpha, batch_t_osz
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

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        x_opt = xp.asarray(self.x_opt)
        R = xp.asarray(self.R)
        Q = xp.asarray(self.Q)

        # z = Q @ Lambda @ R @ (x - x_opt)
        Z = (X - x_opt) @ R.T
        Z = batch_lambda_alpha(Z, 10, self.n_dim)
        Z = Z @ Q.T

        # s = 100 where z * x_opt > 0, else 1
        s = xp.where(Z * x_opt > 0, 100.0, 1.0)
        result = xp.sum((s * Z) ** 2, axis=1)

        # Apply scalar t_osz: need element-wise version
        result = batch_t_osz(result.reshape(-1, 1))[:, 0]
        return result**0.9 + self.f_opt


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

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        x_opt = xp.asarray(self.x_opt)
        R = xp.asarray(self.R)
        Q = xp.asarray(self.Q)

        # z_hat = Lambda @ R @ (x - x_opt)
        Z_hat = (X - x_opt) @ R.T
        Z_hat = batch_lambda_alpha(Z_hat, 10, self.n_dim)

        # Step function
        Z_tilde = xp.where(
            xp.abs(Z_hat) > 0.5,
            xp.floor(0.5 + Z_hat),
            xp.floor(0.5 + 10 * Z_hat) / 10,
        )

        Z = Z_tilde @ Q.T

        # coeffs = 10^(2*i/(n-1))
        i = xp.arange(self.n_dim, dtype=X.dtype)
        if self.n_dim > 1:
            coeffs = xp.power(10.0, 2 * i / (self.n_dim - 1))
        else:
            coeffs = xp.ones(1, dtype=X.dtype)

        # result = 0.1 * max(|z_hat[0]|/1e4, sum(coeffs * z**2))
        term1 = xp.abs(Z_hat[:, 0]) / 1e4
        term2 = xp.sum(coeffs * Z**2, axis=1)
        result = 0.1 * xp.maximum(term1, term2)

        return result + batch_f_pen(X) + self.f_opt


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

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        x_opt = xp.asarray(self.x_opt)
        c = max(1, xp.sqrt(self.n_dim) / 8)

        Z = c * (X - x_opt) + 1

        # Rosenbrock: sum(100*(z_i^2 - z_{i+1})^2 + (z_i - 1)^2)
        z_i = Z[:, :-1]
        z_i1 = Z[:, 1:]
        result = xp.sum(100 * (z_i**2 - z_i1) ** 2 + (z_i - 1) ** 2, axis=1)

        return result + self.f_opt


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

    def _generate_x_opt(self) -> np.ndarray:
        """Compute x_opt such that z = 1 at optimum.

        For Rosenbrock, optimal z = 1 (all ones).
        With z = c * R @ x + 0.5, we need c * R @ x_opt = 0.5.
        Therefore x_opt = R^(-1) @ (0.5/c * ones).
        """
        c = max(1, np.sqrt(self.n_dim) / 8)
        ones = np.ones(self.n_dim)
        # x_opt = R^T @ (0.5/c * ones) since R is orthogonal (R^(-1) = R^T)
        return self.R.T @ (0.5 / c * ones)

    def _create_objective_function(self) -> None:
        c = max(1, np.sqrt(self.n_dim) / 8)

        def rosenbrock_rotated(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            # At x = x_opt: z = c * R @ x_opt + 0.5 = 0.5 + 0.5 = 1 (optimum for Rosenbrock)
            z = c * self.R @ x + 0.5

            result = 0.0
            for i in range(self.n_dim - 1):
                result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2

            return result + self.f_opt

        self.pure_objective_function = rosenbrock_rotated

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        R = xp.asarray(self.R)
        c = max(1, xp.sqrt(self.n_dim) / 8)

        # z = c * R @ x + 0.5 (x_opt is computed to give z = 1 at optimum)
        Z = c * (X @ R.T) + 0.5

        # Rosenbrock: sum(100*(z_i^2 - z_{i+1})^2 + (z_i - 1)^2)
        z_i = Z[:, :-1]
        z_i1 = Z[:, 1:]
        result = xp.sum(100 * (z_i**2 - z_i1) ** 2 + (z_i - 1) ** 2, axis=1)

        return result + self.f_opt
