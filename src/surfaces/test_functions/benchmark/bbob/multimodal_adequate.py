# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""BBOB Multimodal Functions with Adequate Global Structure (f15-f19)."""

import math
from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from .._batch_transforms import batch_f_pen, batch_lambda_alpha, batch_t_asy, batch_t_osz
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
        "eval_cost": 4.8,
        "name": "Rastrigin Function",
        "func_id": 15,
        "unimodal": False,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        Lambda = self.lambda_alpha(10)
        x = self._params_to_array(params)
        z = self.R @ Lambda @ self.Q @ self.t_asy(self.t_osz(self.R @ (x - self.x_opt)), 0.2)
        D = self.n_dim
        return 10 * (D - np.sum(np.cos(2 * np.pi * z))) + np.sum(z**2) + self.f_opt

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        x_opt = xp.asarray(self.x_opt)
        R = xp.asarray(self.R)
        Q = xp.asarray(self.Q)
        D = self.n_dim

        # z = R @ Lambda @ Q @ t_asy(t_osz(R @ (x - x_opt)), 0.2)
        Z = (X - x_opt) @ R.T
        Z = batch_t_osz(Z)
        Z = batch_t_asy(Z, 0.2, self.n_dim)
        Z = Z @ Q.T
        Z = batch_lambda_alpha(Z, 10, self.n_dim)
        Z = Z @ R.T

        cos_sum = xp.sum(xp.cos(2 * math.pi * Z), axis=1)
        sq_sum = xp.sum(Z**2, axis=1)
        return 10 * (D - cos_sum) + sq_sum + self.f_opt


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
        "eval_cost": 16.6,
        "name": "Weierstrass Function",
        "func_id": 16,
        "unimodal": False,
        "separable": False,
        "differentiable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        Lambda = self.lambda_alpha(1.0 / 100)
        k_max = 12
        a = 0.5
        b = 3

        # Precompute the offset: f0 = sum(a^k * cos(2*pi*b^k*0.5))
        f0 = sum(a**k * np.cos(np.pi * b**k) for k in range(k_max))

        x = self._params_to_array(params)
        z = self.R @ Lambda @ self.Q @ self.t_osz(self.R @ (x - self.x_opt))

        D = self.n_dim
        result = 0.0
        for i in range(D):
            for k in range(k_max):
                result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))

        result = 10 * ((result / D - f0) ** 3)
        return result + 10 * self.f_pen(x) / D + self.f_opt

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        x_opt = xp.asarray(self.x_opt)
        R = xp.asarray(self.R)
        Q = xp.asarray(self.Q)
        D = self.n_dim

        k_max = 12
        a = 0.5
        b = 3

        # Precompute offset: f0 = sum(a^k * cos(2*pi*b^k*0.5))
        k = xp.arange(k_max, dtype=X.dtype)
        f0 = xp.sum(a**k * xp.cos(math.pi * b**k))

        # z = R @ Lambda^(1/100) @ Q @ T_osz(R @ (x - x_opt))
        Z = (X - x_opt) @ R.T
        Z = batch_t_osz(Z)
        Z = Z @ Q.T
        Z = batch_lambda_alpha(Z, 1.0 / 100, self.n_dim)
        Z = Z @ R.T

        # Vectorize double loop: sum over i and k
        b_pow_k = b**k  # shape (k_max,)
        a_pow_k = a**k  # shape (k_max,)

        cos_args = 2 * math.pi * (Z[:, :, None] + 0.5) * b_pow_k
        cos_terms = a_pow_k * xp.cos(cos_args)  # (n_points, D, k_max)

        result = xp.sum(cos_terms, axis=(1, 2))  # (n_points,)
        result = 10 * ((result / D - f0) ** 3)

        return result + 10 * batch_f_pen(X) / D + self.f_opt


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
        "eval_cost": 4.1,
        "name": "Schaffer's F7 Function",
        "func_id": 17,
        "unimodal": False,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        Lambda = self.lambda_alpha(10)
        x = self._params_to_array(params)
        z = Lambda @ self.Q @ self.t_asy(self.R @ (x - self.x_opt), 0.5)

        s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        result = np.sum(np.sqrt(s) * (np.sin(50 * s**0.2) ** 2 + 1))
        result = (result / (self.n_dim - 1)) ** 2

        return result + 10 * self.f_pen(x) + self.f_opt

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        x_opt = xp.asarray(self.x_opt)
        R = xp.asarray(self.R)
        Q = xp.asarray(self.Q)

        # z = Lambda @ Q @ t_asy(R @ (x - x_opt), 0.5)
        Z = (X - x_opt) @ R.T
        Z = batch_t_asy(Z, 0.5, self.n_dim)
        Z = Z @ Q.T
        Z = batch_lambda_alpha(Z, 10, self.n_dim)

        # s = sqrt(z[:-1]^2 + z[1:]^2)
        S = xp.sqrt(Z[:, :-1] ** 2 + Z[:, 1:] ** 2)

        # sum(sqrt(s) * (sin(50 * s^0.2)^2 + 1))
        result = xp.sum(xp.sqrt(S) * (xp.sin(50 * S**0.2) ** 2 + 1), axis=1)
        result = (result / (self.n_dim - 1)) ** 2

        return result + 10 * batch_f_pen(X) + self.f_opt


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
        "eval_cost": 4.1,
        "name": "Schaffer's F7 Function, Ill-Conditioned",
        "func_id": 18,
        "unimodal": False,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        Lambda = self.lambda_alpha(1000)
        x = self._params_to_array(params)
        z = Lambda @ self.Q @ self.t_asy(self.R @ (x - self.x_opt), 0.5)

        s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        result = np.sum(np.sqrt(s) * (np.sin(50 * s**0.2) ** 2 + 1))
        result = (result / (self.n_dim - 1)) ** 2

        return result + 10 * self.f_pen(x) + self.f_opt

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        x_opt = xp.asarray(self.x_opt)
        R = xp.asarray(self.R)
        Q = xp.asarray(self.Q)

        # z = Lambda @ Q @ t_asy(R @ (x - x_opt), 0.5)
        Z = (X - x_opt) @ R.T
        Z = batch_t_asy(Z, 0.5, self.n_dim)
        Z = Z @ Q.T
        Z = batch_lambda_alpha(Z, 1000, self.n_dim)  # alpha=1000 for ill-conditioned

        # s = sqrt(z[:-1]^2 + z[1:]^2)
        S = xp.sqrt(Z[:, :-1] ** 2 + Z[:, 1:] ** 2)

        # sum(sqrt(s) * (sin(50 * s^0.2)^2 + 1))
        result = xp.sum(xp.sqrt(S) * (xp.sin(50 * S**0.2) ** 2 + 1), axis=1)
        result = (result / (self.n_dim - 1)) ** 2

        return result + 10 * batch_f_pen(X) + self.f_opt


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
        "eval_cost": 1.9,
        "name": "Composite Griewank-Rosenbrock Function F8F2",
        "func_id": 19,
        "unimodal": False,
        "separable": False,
    }

    def _generate_x_opt(self) -> np.ndarray:
        """Compute x_opt such that z = 1 at optimum.

        Like f9 (RosenbrockRotated), the Rosenbrock part has optimum at z = 1.
        With z = c * R @ x + 0.5, we need c * R @ x_opt = 0.5.
        Therefore x_opt = R^(-1) @ (0.5/c * ones) = R^T @ (0.5/c * ones).
        """
        c = max(1, np.sqrt(self.n_dim) / 8)
        ones = np.ones(self.n_dim)
        return self.R.T @ (0.5 / c * ones)

    def _objective(self, params: Dict[str, Any]) -> float:
        c = max(1, np.sqrt(self.n_dim) / 8)
        x = self._params_to_array(params)
        z = c * self.R @ x + 0.5

        # D-1 Rosenbrock-like terms (no wrap-around per COCO spec)
        result = 0.0
        for i in range(self.n_dim - 1):
            s_i = 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
            result += s_i / 4000 - np.cos(s_i)

        return 10 * result / (self.n_dim - 1) + 10 + self.f_opt

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        R = xp.asarray(self.R)
        D = self.n_dim
        c = max(1, math.sqrt(D) / 8)

        # z = c * R @ x + 0.5 (x_opt is computed to give z = 1 at optimum)
        Z = c * (X @ R.T) + 0.5

        # D-1 consecutive pairs (no wrap-around per COCO spec)
        Z_curr = Z[:, :-1]
        Z_next = Z[:, 1:]
        S = 100 * (Z_curr**2 - Z_next) ** 2 + (Z_curr - 1) ** 2

        result = xp.sum(S / 4000 - xp.cos(S), axis=1)

        return 10 * result / (D - 1) + 10 + self.f_opt
