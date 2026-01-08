# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2008 large-scale benchmark functions F1-F7.

These functions are from the CEC 2008 Special Session on Large Scale Global
Optimization. All functions are 1000-dimensional and shifted.

References
----------
Tang, K., Li, X., Suganthan, P. N., Yang, Z., & Weise, T. (2008).
Benchmark Functions for the CEC'2008 Special Session and Competition
on Large Scale Global Optimization.
Technical Report, Nature Inspired Computation and Applications Laboratory.
"""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2008 import CEC2008SeparableFunction, CEC2008NonSeparableFunction


# =============================================================================
# F1: Shifted Sphere (Separable, Unimodal)
# =============================================================================


class ShiftedSphere2008(CEC2008SeparableFunction):
    """F1: Shifted Sphere Function.

    f(x) = sum((x_i - o_i)^2)

    Properties:
    - Unimodal
    - Separable
    - Scalable
    """

    _spec = {
        **CEC2008SeparableFunction._spec,
        "func_id": 1,
        "name": "ShiftedSphere2008",
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            return np.sum(z**2)

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        return xp.sum(Z**2, axis=1)


# =============================================================================
# F2: Shifted Schwefel 2.21 (Separable, Unimodal)
# =============================================================================


class ShiftedSchwefel221(CEC2008SeparableFunction):
    """F2: Shifted Schwefel's Problem 2.21.

    f(x) = max(|x_i - o_i|)

    Properties:
    - Unimodal
    - Separable
    - Non-differentiable
    """

    _spec = {
        **CEC2008SeparableFunction._spec,
        "func_id": 2,
        "name": "ShiftedSchwefel221",
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
        "differentiable": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            return np.max(np.abs(z))

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        return xp.max(xp.abs(Z), axis=1)


# =============================================================================
# F3: Shifted Rosenbrock (Non-Separable, Multimodal)
# =============================================================================


class ShiftedRosenbrock2008(CEC2008NonSeparableFunction):
    """F3: Shifted Rosenbrock's Function.

    f(x) = sum(100 * (z_{i+1} - z_i^2)^2 + (z_i - 1)^2)

    Properties:
    - Multimodal (for high dimensions)
    - Non-separable
    - Has a narrow valley leading to the optimum
    """

    _spec = {
        **CEC2008NonSeparableFunction._spec,
        "func_id": 3,
        "name": "ShiftedRosenbrock2008",
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            result = 0.0
            for i in range(len(z) - 1):
                result += 100 * (z[i + 1] - z[i]**2)**2 + (z[i] - 1)**2
            return result

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        term1 = 100 * (Z[:, 1:] - Z[:, :-1]**2)**2
        term2 = (Z[:, :-1] - 1)**2
        return xp.sum(term1 + term2, axis=1)


# =============================================================================
# F4: Shifted Rastrigin (Separable, Multimodal)
# =============================================================================


class ShiftedRastrigin2008(CEC2008SeparableFunction):
    """F4: Shifted Rastrigin's Function.

    f(x) = sum(z_i^2 - 10*cos(2*pi*z_i) + 10)

    Properties:
    - Highly multimodal (~10^D local minima)
    - Separable
    - Regular distribution of local minima
    """

    _spec = {
        **CEC2008SeparableFunction._spec,
        "func_id": 4,
        "name": "ShiftedRastrigin2008",
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            return np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        return xp.sum(Z**2 - 10 * xp.cos(2 * np.pi * Z) + 10, axis=1)


# =============================================================================
# F5: Shifted Griewank (Non-Separable, Multimodal)
# =============================================================================


class ShiftedGriewank2008(CEC2008NonSeparableFunction):
    """F5: Shifted Griewank's Function.

    f(x) = 1 + sum(z_i^2)/4000 - prod(cos(z_i/sqrt(i)))

    Properties:
    - Multimodal
    - Non-separable due to product term
    - Has many regularly distributed local minima
    """

    _spec = {
        **CEC2008NonSeparableFunction._spec,
        "func_id": 5,
        "name": "ShiftedGriewank2008",
        "default_bounds": (-600.0, 600.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            i = np.arange(1, len(z) + 1)
            sum_term = np.sum(z**2) / 4000
            prod_term = np.prod(np.cos(z / np.sqrt(i)))
            return 1 + sum_term - prod_term

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        i = xp.arange(1, Z.shape[1] + 1, dtype=X.dtype)
        sum_term = xp.sum(Z**2, axis=1) / 4000
        prod_term = xp.prod(xp.cos(Z / xp.sqrt(i)), axis=1)
        return 1 + sum_term - prod_term


# =============================================================================
# F6: Shifted Ackley (Separable, Multimodal)
# =============================================================================


class ShiftedAckley2008(CEC2008SeparableFunction):
    """F6: Shifted Ackley's Function.

    f(x) = -20*exp(-0.2*sqrt(sum(z_i^2)/D)) - exp(sum(cos(2*pi*z_i))/D) + 20 + e

    Properties:
    - Multimodal
    - Separable
    - Nearly flat outer region with central deep hole
    """

    _spec = {
        **CEC2008SeparableFunction._spec,
        "func_id": 6,
        "name": "ShiftedAckley2008",
        "default_bounds": (-32.0, 32.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            D = len(z)
            term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / D))
            term2 = -np.exp(np.sum(np.cos(2 * np.pi * z)) / D)
            return term1 + term2 + 20 + np.e

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        D = Z.shape[1]
        term1 = -20 * xp.exp(-0.2 * xp.sqrt(xp.sum(Z**2, axis=1) / D))
        term2 = -xp.exp(xp.sum(xp.cos(2 * np.pi * Z), axis=1) / D)
        return term1 + term2 + 20 + np.e


# =============================================================================
# F7: Fast Fractal Double Dip (Non-Separable, Multimodal)
# =============================================================================


class FastFractalDoubleDip(CEC2008NonSeparableFunction):
    """F7: Fast Fractal "Double Dip" Function.

    This is a complex fractal function with multiple scales of structure.
    It uses a recursive doublet structure that creates a self-similar
    landscape at multiple scales.

    Properties:
    - Highly multimodal
    - Non-separable
    - Fractal structure
    """

    _spec = {
        **CEC2008NonSeparableFunction._spec,
        "func_id": 7,
        "name": "FastFractalDoubleDip",
        "default_bounds": (-1.0, 1.0),
        "unimodal": False,
    }

    @staticmethod
    def _fractal_1d(x: float, depth: int = 3) -> float:
        """Compute fractal function for 1D."""
        result = 0.0
        for k in range(1, depth + 1):
            scale = 2**(k - 1)
            result += np.sin(scale * np.pi * x) / scale
        return result

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            D = len(z)
            result = 0.0
            for i in range(D):
                # Create doublet structure by combining adjacent dimensions
                if i < D - 1:
                    xi = z[i]
                    xj = z[i + 1]
                    # Double dip contribution
                    result += self._fractal_1d(xi) * self._fractal_1d(xj)
                result += z[i]**2
            return result

        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        D = Z.shape[1]

        # Simplified batch implementation
        # Sum of squares term
        result = xp.sum(Z**2, axis=1)

        # Fractal coupling term (simplified)
        for k in range(1, 4):  # depth=3
            scale = 2**(k - 1)
            sin_term = xp.sin(scale * np.pi * Z)
            # Coupling between adjacent dimensions
            coupling = xp.sum(sin_term[:, :-1] * sin_term[:, 1:], axis=1) / scale**2
            result = result + coupling

        return result


# =============================================================================
# All CEC 2008 functions
# =============================================================================

CEC2008_ALL = [
    ShiftedSphere2008,
    ShiftedSchwefel221,
    ShiftedRosenbrock2008,
    ShiftedRastrigin2008,
    ShiftedGriewank2008,
    ShiftedAckley2008,
    FastFractalDoubleDip,
]

CEC2008_SEPARABLE = [
    ShiftedSphere2008,
    ShiftedSchwefel221,
    ShiftedRastrigin2008,
    ShiftedAckley2008,
]

CEC2008_NONSEPARABLE = [
    ShiftedRosenbrock2008,
    ShiftedGriewank2008,
    FastFractalDoubleDip,
]
