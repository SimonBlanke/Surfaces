# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""BBOB Multimodal Functions with Weak Global Structure (f20-f24)."""

import math
from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from .._batch_transforms import batch_f_pen, batch_lambda_alpha, batch_t_osz
from ._base_bbob import BBOBFunction


class Schwefel(BBOBFunction):
    """f20: Schwefel Function.

    Deceptive function where the global optimum is far from next best.
    Has many deep local optima.

    Properties:
    - Highly multimodal
    - Non-separable
    - Deceptive (second-best regions far from optimum)
    """

    _spec = {
        "name": "Schwefel Function",
        "func_id": 20,
        "unimodal": False,
        "separable": False,
    }

    def _generate_x_opt(self) -> np.ndarray:
        """Generate x_opt with special structure for Schwefel.

        Per COCO: x_opt = 4.2096874633/2 * sign_vector
        where sign_vector is randomly +1 or -1 per dimension.
        """
        sign_vector = np.where(self._rng.rand(self.n_dim) > 0.5, 1, -1)
        return 4.2096874633 / 2 * sign_vector

    def _create_objective_function(self) -> None:
        Lambda = self.lambda_alpha(10)
        abs_x_opt = np.abs(self.x_opt)  # = 4.2096874633/2 for all dimensions

        def schwefel(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            D = self.n_dim

            # Step 1: x_hat = 2 * sign(x_opt) * x
            x_hat = 2 * np.sign(self.x_opt) * x

            # Step 2: Sequential coupling (makes function non-separable)
            # z_hat[0] = x_hat[0]
            # z_hat[i+1] = x_hat[i+1] + 0.25 * (x_hat[i] - 2*|x_opt[i]|)
            z_hat = np.zeros(D)
            z_hat[0] = x_hat[0]
            for i in range(D - 1):
                z_hat[i + 1] = x_hat[i + 1] + 0.25 * (x_hat[i] - 2 * abs_x_opt[i])

            # Step 3: z = 100 * (Lambda @ (z_hat - 2*|x_opt|) + 2*|x_opt|)
            z = 100 * (Lambda @ (z_hat - 2 * abs_x_opt) + 2 * abs_x_opt)

            # Schwefel function: -1/(100*D) * sum(z * sin(sqrt(|z|)))
            result = 0.0
            for i in range(D):
                zi = z[i]
                if np.abs(zi) <= 500:
                    result += zi * np.sin(np.sqrt(np.abs(zi)))
                else:
                    # Penalty for out-of-bounds
                    result += (500 - np.abs(zi) % 500) * np.sin(
                        np.sqrt(np.abs(500 - np.abs(zi) % 500))
                    )
                    result -= (zi - 500) ** 2 / (10000 * D) * np.sign(zi)

            result = -result / (100 * D)
            # Penalty applies to z/100 per COCO, but simplified here
            return result + 4.189828872724339 + 100 * self.f_pen(z / 100) + self.f_opt

        self.pure_objective_function = schwefel

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        x_opt = xp.asarray(self.x_opt)
        D = self.n_dim
        abs_x_opt = xp.abs(x_opt)

        # Step 1: x_hat = 2 * sign(x_opt) * x
        X_hat = 2 * xp.sign(x_opt) * X  # (n_points, D)

        # Step 2: Sequential coupling (vectorized)
        # z_hat[0] = x_hat[0]
        # z_hat[i+1] = x_hat[i+1] + 0.25 * (x_hat[i] - 2*|x_opt[i]|)
        Z_hat = xp.zeros_like(X_hat)
        Z_hat[:, 0] = X_hat[:, 0]
        for i in range(D - 1):
            Z_hat[:, i + 1] = X_hat[:, i + 1] + 0.25 * (X_hat[:, i] - 2 * abs_x_opt[i])

        # Step 3: z = 100 * (Lambda @ (z_hat - 2*|x_opt|) + 2*|x_opt|)
        Z = batch_lambda_alpha(Z_hat - 2 * abs_x_opt, 10, D) + 2 * abs_x_opt
        Z = 100 * Z

        # Conditional computation vectorized
        abs_Z = xp.abs(Z)
        in_bounds = abs_Z <= 500

        # In-bounds case: z * sin(sqrt(|z|))
        term_in = Z * xp.sin(xp.sqrt(abs_Z))

        # Out-of-bounds case
        mod_term = 500 - abs_Z % 500
        term_out = mod_term * xp.sin(xp.sqrt(xp.abs(mod_term)))
        term_out = term_out - (Z - 500) ** 2 / (10000 * D) * xp.sign(Z)

        # Combine using where
        terms = xp.where(in_bounds, term_in, term_out)
        result = -xp.sum(terms, axis=1) / (100 * D)

        # Penalty applies to z/100 per COCO
        return result + 4.189828872724339 + 100 * batch_f_pen(Z / 100) + self.f_opt


class Gallagher101(BBOBFunction):
    """f21: Gallagher's Gaussian 101-me Peaks Function.

    Function with 101 Gaussian peaks of different heights.
    One global peak and 100 local peaks.

    Properties:
    - Highly multimodal (101 peaks)
    - Non-separable
    - Gaussian peaks
    """

    _spec = {
        "name": "Gallagher's Gaussian 101-me Peaks Function",
        "func_id": 21,
        "unimodal": False,
        "separable": False,
    }

    def __init__(
        self,
        n_dim=10,
        instance=1,
        objective="minimize",
        modifiers=None,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
    ):
        super().__init__(
            n_dim, instance, objective, modifiers, memory, collect_data, callbacks, catch_errors
        )
        self._setup_peaks(101)

    def _setup_peaks(self, n_peaks: int):
        """Set up the Gaussian peak parameters."""
        # Weights for the peaks
        self._w = np.hstack([10, 1.1 + 8 * np.arange(n_peaks - 1) / (n_peaks - 2)])

        # Alpha values for conditioning
        alpha_base = np.power(1000, 2 * np.arange(n_peaks - 1) / (n_peaks - 2))
        self._alpha = np.hstack([1000**2, self._rng.permutation(alpha_base)])

        # Peak locations
        self._y = np.vstack([self.x_opt, self._rng.uniform(-4.9, 4.9, (n_peaks - 1, self.n_dim))])

        # Condition matrices
        self._C = []
        for k in range(n_peaks):
            i = np.arange(self.n_dim)
            diag = (
                np.power(self._alpha[k], 0.5 * i / (self.n_dim - 1))
                if self.n_dim > 1
                else np.ones(1)
            )
            C = np.diag(diag / np.power(self._alpha[k], 0.25))
            self._C.append(self.R.T @ C @ self.R)

    def _create_objective_function(self) -> None:
        def gallagher(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)

            max_val = 0
            for k in range(len(self._w)):
                diff = x - self._y[k]
                exponent = -0.5 / self.n_dim * diff @ self._C[k] @ diff
                val = self._w[k] * np.exp(exponent)
                max_val = max(max_val, val)

            result = self.t_osz(10 - max_val) ** 2
            return result + self.f_pen(x) + self.f_opt

        self.pure_objective_function = gallagher

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        n_peaks = len(self._w)
        D = self.n_dim

        # Compute all peak values for all points
        peak_vals = xp.zeros((n_points, n_peaks), dtype=X.dtype)
        for k in range(n_peaks):
            y_k = xp.asarray(self._y[k])
            C_k = xp.asarray(self._C[k])
            diff = X - y_k  # (n_points, n_dim)
            # Quadratic form: sum((diff @ C) * diff, axis=1)
            quad = xp.sum((diff @ C_k) * diff, axis=1)  # (n_points,)
            exponent = -0.5 / D * quad
            peak_vals[:, k] = self._w[k] * xp.exp(exponent)

        max_vals = xp.max(peak_vals, axis=1)  # (n_points,)

        # Apply scalar t_osz via batch version
        result = batch_t_osz((10 - max_vals).reshape(-1, 1))[:, 0] ** 2

        return result + batch_f_pen(X) + self.f_opt


class Gallagher21(BBOBFunction):
    """f22: Gallagher's Gaussian 21-hi Peaks Function.

    Function with 21 Gaussian peaks of different heights.
    Similar to f21 but with fewer, higher peaks.

    Properties:
    - Multimodal (21 peaks)
    - Non-separable
    - Gaussian peaks (higher)
    """

    _spec = {
        "name": "Gallagher's Gaussian 21-hi Peaks Function",
        "func_id": 22,
        "unimodal": False,
        "separable": False,
    }

    def __init__(
        self,
        n_dim=10,
        instance=1,
        objective="minimize",
        modifiers=None,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
    ):
        super().__init__(
            n_dim, instance, objective, modifiers, memory, collect_data, callbacks, catch_errors
        )
        self._setup_peaks(21)

    def _setup_peaks(self, n_peaks: int):
        """Set up the Gaussian peak parameters."""
        # Weights for the peaks
        self._w = np.hstack([10, 1.1 + 8 * np.arange(n_peaks - 1) / (n_peaks - 2)])

        # Alpha values for conditioning (higher for f22)
        alpha_base = np.power(1000, 2 * np.arange(n_peaks - 1) / (n_peaks - 2))
        self._alpha = np.hstack([1000**2, self._rng.permutation(alpha_base)])

        # Peak locations (narrower range for f22)
        self._y = np.vstack([self.x_opt, self._rng.uniform(-4.0, 4.0, (n_peaks - 1, self.n_dim))])

        # Condition matrices
        self._C = []
        for k in range(n_peaks):
            i = np.arange(self.n_dim)
            diag = (
                np.power(self._alpha[k], 0.5 * i / (self.n_dim - 1))
                if self.n_dim > 1
                else np.ones(1)
            )
            C = np.diag(diag / np.power(self._alpha[k], 0.25))
            self._C.append(self.R.T @ C @ self.R)

    def _create_objective_function(self) -> None:
        def gallagher(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)

            max_val = 0
            for k in range(len(self._w)):
                diff = x - self._y[k]
                exponent = -0.5 / self.n_dim * diff @ self._C[k] @ diff
                val = self._w[k] * np.exp(exponent)
                max_val = max(max_val, val)

            result = self.t_osz(10 - max_val) ** 2
            return result + self.f_pen(x) + self.f_opt

        self.pure_objective_function = gallagher

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        n_peaks = len(self._w)
        D = self.n_dim

        # Compute all peak values for all points
        peak_vals = xp.zeros((n_points, n_peaks), dtype=X.dtype)
        for k in range(n_peaks):
            y_k = xp.asarray(self._y[k])
            C_k = xp.asarray(self._C[k])
            diff = X - y_k  # (n_points, n_dim)
            # Quadratic form: sum((diff @ C) * diff, axis=1)
            quad = xp.sum((diff @ C_k) * diff, axis=1)  # (n_points,)
            exponent = -0.5 / D * quad
            peak_vals[:, k] = self._w[k] * xp.exp(exponent)

        max_vals = xp.max(peak_vals, axis=1)  # (n_points,)

        # Apply scalar t_osz via batch version
        result = batch_t_osz((10 - max_vals).reshape(-1, 1))[:, 0] ** 2

        return result + batch_f_pen(X) + self.f_opt


class Katsuura(BBOBFunction):
    """f23: Katsuura Function.

    Continuous but highly rugged function.
    Has extremely many local optima.

    Properties:
    - Highly multimodal
    - Non-separable
    - Continuous
    - Extremely rugged
    """

    _spec = {
        "name": "Katsuura Function",
        "func_id": 23,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        Lambda = self.lambda_alpha(100)

        def katsuura(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self.Q @ Lambda @ self.R @ (x - self.x_opt)

            D = self.n_dim
            result = 1.0

            for i in range(D):
                inner_sum = 0.0
                for j in range(1, 33):
                    inner_sum += np.abs(2**j * z[i] - np.round(2**j * z[i])) / (2**j)
                result *= (1 + (i + 1) * inner_sum) ** (10 / D**1.2)

            result = 10 / (D**2) * result - 10 / (D**2)
            return result + self.f_pen(x) + self.f_opt

        self.pure_objective_function = katsuura

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        x_opt = xp.asarray(self.x_opt)
        R = xp.asarray(self.R)
        Q = xp.asarray(self.Q)
        D = self.n_dim

        # z = Q @ Lambda @ R @ (x - x_opt)
        Z = (X - x_opt) @ R.T
        Z = batch_lambda_alpha(Z, 100, D)
        Z = Z @ Q.T

        # Vectorize double loop over i (dimensions) and j (1..32)
        j = xp.arange(1, 33, dtype=X.dtype)  # shape (32,)
        pow2j = xp.power(2.0, j)  # shape (32,)

        # Z[:, :, None] has shape (n_points, D, 1)
        # Broadcast to (n_points, D, 32)
        scaled = Z[:, :, None] * pow2j
        inner = xp.abs(scaled - xp.round(scaled)) / pow2j  # (n_points, D, 32)
        inner_sum = xp.sum(inner, axis=2)  # (n_points, D)

        # (i+1) factor for each dimension
        i_plus_1 = xp.arange(1, D + 1, dtype=X.dtype)  # (D,)

        # (1 + (i+1) * inner_sum) ** (10 / D^1.2)
        terms = xp.power(1 + i_plus_1 * inner_sum, 10 / D**1.2)  # (n_points, D)
        result = xp.prod(terms, axis=1)  # (n_points,)

        result = 10 / D**2 * result - 10 / D**2
        return result + batch_f_pen(X) + self.f_opt


class LunacekBiRastrigin(BBOBFunction):
    """f24: Lunacek Bi-Rastrigin Function.

    Bi-modal function combining two Rastrigin functions.
    Has a deceptive global structure.

    Properties:
    - Bi-modal at global scale
    - Highly multimodal at local scale
    - Non-separable
    - Deceptive
    """

    _spec = {
        "name": "Lunacek Bi-Rastrigin Function",
        "func_id": 24,
        "unimodal": False,
        "separable": False,
    }

    def _generate_x_opt(self) -> np.ndarray:
        """Generate x_opt with special structure."""
        return 0.5 * 2.5 * np.where(self._rng.rand(self.n_dim) > 0.5, 1, -1)

    def _create_objective_function(self) -> None:
        mu0 = 2.5
        D = self.n_dim
        s = 1 - 1 / (2 * np.sqrt(D + 20) - 8.2)
        mu1 = -np.sqrt((mu0**2 - 1) / s)

        Lambda = self.lambda_alpha(100)

        def lunacek_bi_rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)

            # Transform x
            x_hat = 2 * np.sign(self.x_opt) * x
            x_tilde = x_hat - mu0

            # Sum terms
            sum1 = np.sum((x_tilde) ** 2)
            sum2 = D + s * np.sum((x_hat - mu1) ** 2)

            # Apply rotation for cosine term
            z = self.Q @ Lambda @ self.R @ x_tilde
            sum3 = np.sum(np.cos(2 * np.pi * z))

            result = min(sum1, sum2) + 10 * (D - sum3)
            return result + 1e4 * self.f_pen(x) + self.f_opt

        self.pure_objective_function = lunacek_bi_rastrigin

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        x_opt = xp.asarray(self.x_opt)
        R = xp.asarray(self.R)
        Q = xp.asarray(self.Q)
        D = self.n_dim

        mu0 = 2.5
        s = 1 - 1 / (2 * math.sqrt(D + 20) - 8.2)
        mu1 = -math.sqrt((mu0**2 - 1) / s)

        # Transform x
        X_hat = 2 * xp.sign(x_opt) * X  # (n_points, D)
        X_tilde = X_hat - mu0  # (n_points, D)

        # Sum terms
        sum1 = xp.sum(X_tilde**2, axis=1)  # (n_points,)
        sum2 = D + s * xp.sum((X_hat - mu1) ** 2, axis=1)  # (n_points,)

        # Apply rotation for cosine term: z = Q @ Lambda @ R @ x_tilde
        Z = X_tilde @ R.T
        Z = batch_lambda_alpha(Z, 100, D)
        Z = Z @ Q.T
        sum3 = xp.sum(xp.cos(2 * math.pi * Z), axis=1)

        result = xp.minimum(sum1, sum2) + 10 * (D - sum3)
        return result + 1e4 * batch_f_pen(X) + self.f_opt
