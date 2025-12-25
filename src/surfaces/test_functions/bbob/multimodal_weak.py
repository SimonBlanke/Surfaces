# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""BBOB Multimodal Functions with Weak Global Structure (f20-f24)."""

from typing import Any, Dict

import numpy as np

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
        """Generate x_opt with special structure for Schwefel."""
        x = self._rng.uniform(-4, 4, self.n_dim)
        # Make half of the components have same sign
        x[::2] = np.abs(x[::2])
        x[1::2] = -np.abs(x[1::2])
        return 0.5 * 4.2096874633 * x

    def _create_objective_function(self) -> None:
        Lambda = self.lambda_alpha(10)

        def schwefel(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = 2 * np.sign(self.x_opt) * (x - self.x_opt)
            z = Lambda @ (z - 2 * np.abs(self.x_opt)) + 2 * np.abs(self.x_opt)
            z = 100 * (z + 4.2096874633 * np.abs(self.x_opt) / 2)

            D = self.n_dim
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
            return result + 4.189828872724339 + self.f_pen(x) + self.f_opt

        self.pure_objective_function = schwefel


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
        sleep=0,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
        noise=None,
    ):
        super().__init__(
            n_dim, instance, objective, sleep, memory, collect_data, callbacks, catch_errors, noise
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
        sleep=0,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
        noise=None,
    ):
        super().__init__(
            n_dim, instance, objective, sleep, memory, collect_data, callbacks, catch_errors, noise
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
            sum3 = 10 * np.sum(np.cos(2 * np.pi * z))

            result = min(sum1, sum2) + 10 * (D - sum3 / D)
            return result + 1e4 * self.f_pen(x) + self.f_opt

        self.pure_objective_function = lunacek_bi_rastrigin
