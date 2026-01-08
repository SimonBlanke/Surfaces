# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2010 large-scale benchmark functions F1-F20.

These functions are from the CEC 2010 Special Session on Large Scale Global
Optimization. All functions have 1000 dimensions with partial separability.

References
----------
Tang, K., Li, X., Suganthan, P. N., Yang, Z., & Weise, T. (2010).
Benchmark Functions for the CEC'2010 Special Session and Competition
on Large Scale Global Optimization.
Technical Report, Nature Inspired Computation and Applications Laboratory.
"""

from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2010 import (
    CEC2010SeparableFunction,
    CEC2010PartialSeparableFunction,
    CEC2010NonSeparableFunction,
    CEC2010CompositionFunction,
)


# =============================================================================
# Helper functions (basic benchmark functions)
# =============================================================================


def _elliptic(z: np.ndarray) -> float:
    """Elliptic function."""
    D = len(z)
    i = np.arange(D)
    return np.sum(10 ** (6 * i / (D - 1)) * z**2)


def _rastrigin(z: np.ndarray) -> float:
    """Rastrigin function."""
    return np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)


def _ackley(z: np.ndarray) -> float:
    """Ackley function."""
    D = len(z)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / D))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * z)) / D)
    return term1 + term2 + 20 + np.e


def _schwefel(z: np.ndarray) -> float:
    """Schwefel 1.2 function."""
    D = len(z)
    result = 0.0
    for i in range(D):
        result += np.sum(z[:i + 1])**2
    return result


def _rosenbrock(z: np.ndarray) -> float:
    """Rosenbrock function."""
    return np.sum(100 * (z[1:] - z[:-1]**2)**2 + (z[:-1] - 1)**2)


def _griewank(z: np.ndarray) -> float:
    """Griewank function."""
    i = np.arange(1, len(z) + 1)
    return np.sum(z**2) / 4000 - np.prod(np.cos(z / np.sqrt(i))) + 1


# =============================================================================
# F1-F3: Fully Separable Functions
# =============================================================================


class SeparableElliptic(CEC2010SeparableFunction):
    """F1: Separable Elliptic Function."""

    _spec = {
        **CEC2010SeparableFunction._spec,
        "func_id": 1,
        "name": "SeparableElliptic",
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            return _elliptic(z)
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        D = Z.shape[1]
        i = xp.arange(D, dtype=X.dtype)
        weights = 10 ** (6 * i / (D - 1))
        return xp.sum(weights * Z**2, axis=1)


class SeparableRastrigin(CEC2010SeparableFunction):
    """F2: Separable Rastrigin Function."""

    _spec = {
        **CEC2010SeparableFunction._spec,
        "func_id": 2,
        "name": "SeparableRastrigin",
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            return _rastrigin(z)
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        return xp.sum(Z**2 - 10 * xp.cos(2 * np.pi * Z) + 10, axis=1)


class SeparableAckley(CEC2010SeparableFunction):
    """F3: Separable Ackley Function."""

    _spec = {
        **CEC2010SeparableFunction._spec,
        "func_id": 3,
        "name": "SeparableAckley",
        "default_bounds": (-32.0, 32.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            return _ackley(z)
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        D = Z.shape[1]
        term1 = -20 * xp.exp(-0.2 * xp.sqrt(xp.sum(Z**2, axis=1) / D))
        term2 = -xp.exp(xp.sum(xp.cos(2 * np.pi * Z), axis=1) / D)
        return term1 + term2 + 20 + np.e


# =============================================================================
# F4-F7: Single-group Rotated Functions
# =============================================================================


class SingleGroupElliptic(CEC2010PartialSeparableFunction):
    """F4: Single-group Shifted and m-rotated Elliptic Function."""

    _spec = {
        **CEC2010PartialSeparableFunction._spec,
        "func_id": 4,
        "name": "SingleGroupElliptic",
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            # First m variables are rotated
            z_rot = self._get_group_rotation(0) @ z[:self.m]
            return _elliptic(z_rot) * 1e6 + _elliptic(z[self.m:])
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        # Apply permutation
        perm = self._get_permutation()
        Z = Z[:, perm]
        D = Z.shape[1]
        m = self.m

        # First group (rotated)
        R = xp.asarray(self._get_group_rotation(0))
        Z_rot = Z[:, :m] @ R.T
        i1 = xp.arange(m, dtype=X.dtype)
        w1 = 10 ** (6 * i1 / (m - 1))
        term1 = xp.sum(w1 * Z_rot**2, axis=1) * 1e6

        # Remaining variables
        i2 = xp.arange(D - m, dtype=X.dtype)
        w2 = 10 ** (6 * i2 / (D - m - 1)) if D - m > 1 else xp.ones(D - m)
        term2 = xp.sum(w2 * Z[:, m:]**2, axis=1)

        return term1 + term2


class SingleGroupRastrigin(CEC2010PartialSeparableFunction):
    """F5: Single-group Shifted and m-rotated Rastrigin Function."""

    _spec = {
        **CEC2010PartialSeparableFunction._spec,
        "func_id": 5,
        "name": "SingleGroupRastrigin",
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            z_rot = self._get_group_rotation(0) @ z[:self.m]
            return _rastrigin(z_rot) * 1e6 + _rastrigin(z[self.m:])
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        m = self.m

        R = xp.asarray(self._get_group_rotation(0))
        Z_rot = Z[:, :m] @ R.T
        term1 = xp.sum(Z_rot**2 - 10 * xp.cos(2 * np.pi * Z_rot) + 10, axis=1) * 1e6
        term2 = xp.sum(Z[:, m:]**2 - 10 * xp.cos(2 * np.pi * Z[:, m:]) + 10, axis=1)

        return term1 + term2


class SingleGroupAckley(CEC2010PartialSeparableFunction):
    """F6: Single-group Shifted and m-rotated Ackley Function."""

    _spec = {
        **CEC2010PartialSeparableFunction._spec,
        "func_id": 6,
        "name": "SingleGroupAckley",
        "default_bounds": (-32.0, 32.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            z_rot = self._get_group_rotation(0) @ z[:self.m]
            return _ackley(z_rot) * 1e6 + _ackley(z[self.m:])
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        m = self.m

        R = xp.asarray(self._get_group_rotation(0))
        Z_rot = Z[:, :m] @ R.T
        term1_1 = -20 * xp.exp(-0.2 * xp.sqrt(xp.sum(Z_rot**2, axis=1) / m))
        term1_2 = -xp.exp(xp.sum(xp.cos(2 * np.pi * Z_rot), axis=1) / m)
        term1 = (term1_1 + term1_2 + 20 + np.e) * 1e6

        D_rem = X.shape[1] - m
        term2_1 = -20 * xp.exp(-0.2 * xp.sqrt(xp.sum(Z[:, m:]**2, axis=1) / D_rem))
        term2_2 = -xp.exp(xp.sum(xp.cos(2 * np.pi * Z[:, m:]), axis=1) / D_rem)
        term2 = term2_1 + term2_2 + 20 + np.e

        return term1 + term2


class SingleGroupSchwefel(CEC2010PartialSeparableFunction):
    """F7: Single-group Shifted m-dimensional Schwefel Function."""

    _spec = {
        **CEC2010PartialSeparableFunction._spec,
        "func_id": 7,
        "name": "SingleGroupSchwefel",
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            z_rot = self._get_group_rotation(0) @ z[:self.m]
            return _schwefel(z_rot) * 1e6 + _elliptic(z[self.m:])
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        m = self.m

        R = xp.asarray(self._get_group_rotation(0))
        Z_rot = Z[:, :m] @ R.T

        # Schwefel for rotated part (vectorized approximation)
        n = X.shape[0]
        term1 = xp.zeros(n)
        for i in range(m):
            term1 = term1 + xp.sum(Z_rot[:, :i + 1], axis=1)**2
        term1 = term1 * 1e6

        # Elliptic for remaining
        D_rem = X.shape[1] - m
        i2 = xp.arange(D_rem, dtype=X.dtype)
        w2 = 10 ** (6 * i2 / (D_rem - 1)) if D_rem > 1 else xp.ones(D_rem)
        term2 = xp.sum(w2 * Z[:, m:]**2, axis=1)

        return term1 + term2


# =============================================================================
# F8-F13: D/m-group Rotated Functions (20 groups of m=50)
# =============================================================================


class MultiGroupElliptic(CEC2010PartialSeparableFunction):
    """F8: D/m-group Shifted and m-rotated Elliptic Function."""

    _spec = {
        **CEC2010PartialSeparableFunction._spec,
        "func_id": 8,
        "name": "MultiGroupElliptic",
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            result = 0.0
            for g in range(self.n_groups):
                z_g = self._get_group(z, g)
                R = self._get_group_rotation(g)
                z_rot = R @ z_g
                result += _elliptic(z_rot)
            return result
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        m = self.m

        result = xp.zeros(X.shape[0])
        i_g = xp.arange(m, dtype=X.dtype)
        w = 10 ** (6 * i_g / (m - 1))

        for g in range(self.n_groups):
            start = g * m
            end = start + m
            R = xp.asarray(self._get_group_rotation(g))
            Z_rot = Z[:, start:end] @ R.T
            result = result + xp.sum(w * Z_rot**2, axis=1)

        return result


class MultiGroupRastrigin(CEC2010PartialSeparableFunction):
    """F9: D/m-group Shifted and m-rotated Rastrigin Function."""

    _spec = {
        **CEC2010PartialSeparableFunction._spec,
        "func_id": 9,
        "name": "MultiGroupRastrigin",
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            result = 0.0
            for g in range(self.n_groups):
                z_g = self._get_group(z, g)
                R = self._get_group_rotation(g)
                z_rot = R @ z_g
                result += _rastrigin(z_rot)
            return result
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        m = self.m

        result = xp.zeros(X.shape[0])
        for g in range(self.n_groups):
            start = g * m
            end = start + m
            R = xp.asarray(self._get_group_rotation(g))
            Z_rot = Z[:, start:end] @ R.T
            result = result + xp.sum(Z_rot**2 - 10 * xp.cos(2 * np.pi * Z_rot) + 10, axis=1)

        return result


class MultiGroupAckley(CEC2010PartialSeparableFunction):
    """F10: D/m-group Shifted and m-rotated Ackley Function."""

    _spec = {
        **CEC2010PartialSeparableFunction._spec,
        "func_id": 10,
        "name": "MultiGroupAckley",
        "default_bounds": (-32.0, 32.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            result = 0.0
            for g in range(self.n_groups):
                z_g = self._get_group(z, g)
                R = self._get_group_rotation(g)
                z_rot = R @ z_g
                result += _ackley(z_rot)
            return result
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        m = self.m

        result = xp.zeros(X.shape[0])
        for g in range(self.n_groups):
            start = g * m
            end = start + m
            R = xp.asarray(self._get_group_rotation(g))
            Z_rot = Z[:, start:end] @ R.T
            t1 = -20 * xp.exp(-0.2 * xp.sqrt(xp.sum(Z_rot**2, axis=1) / m))
            t2 = -xp.exp(xp.sum(xp.cos(2 * np.pi * Z_rot), axis=1) / m)
            result = result + t1 + t2 + 20 + np.e

        return result


class MultiGroupSchwefel(CEC2010PartialSeparableFunction):
    """F11: D/m-group Shifted and m-rotated Schwefel Function."""

    _spec = {
        **CEC2010PartialSeparableFunction._spec,
        "func_id": 11,
        "name": "MultiGroupSchwefel",
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            result = 0.0
            for g in range(self.n_groups):
                z_g = self._get_group(z, g)
                R = self._get_group_rotation(g)
                z_rot = R @ z_g
                result += _schwefel(z_rot)
            return result
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        m = self.m

        result = xp.zeros(X.shape[0])
        for g in range(self.n_groups):
            start = g * m
            end = start + m
            R = xp.asarray(self._get_group_rotation(g))
            Z_rot = Z[:, start:end] @ R.T
            # Schwefel
            for i in range(m):
                result = result + xp.sum(Z_rot[:, :i + 1], axis=1)**2

        return result


class MultiGroupRosenbrock(CEC2010PartialSeparableFunction):
    """F12: D/m-group Shifted and m-rotated Rosenbrock Function."""

    _spec = {
        **CEC2010PartialSeparableFunction._spec,
        "func_id": 12,
        "name": "MultiGroupRosenbrock",
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            result = 0.0
            for g in range(self.n_groups):
                z_g = self._get_group(z, g)
                R = self._get_group_rotation(g)
                z_rot = R @ z_g
                result += _rosenbrock(z_rot)
            return result
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        m = self.m

        result = xp.zeros(X.shape[0])
        for g in range(self.n_groups):
            start = g * m
            end = start + m
            R = xp.asarray(self._get_group_rotation(g))
            Z_rot = Z[:, start:end] @ R.T
            t1 = 100 * (Z_rot[:, 1:] - Z_rot[:, :-1]**2)**2
            t2 = (Z_rot[:, :-1] - 1)**2
            result = result + xp.sum(t1 + t2, axis=1)

        return result


class MultiGroupGriewank(CEC2010PartialSeparableFunction):
    """F13: D/m-group Shifted and m-rotated Griewank Function."""

    _spec = {
        **CEC2010PartialSeparableFunction._spec,
        "func_id": 13,
        "name": "MultiGroupGriewank",
        "default_bounds": (-600.0, 600.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            result = 0.0
            for g in range(self.n_groups):
                z_g = self._get_group(z, g)
                R = self._get_group_rotation(g)
                z_rot = R @ z_g
                result += _griewank(z_rot)
            return result
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        m = self.m

        result = xp.zeros(X.shape[0])
        i_g = xp.arange(1, m + 1, dtype=X.dtype)
        for g in range(self.n_groups):
            start = g * m
            end = start + m
            R = xp.asarray(self._get_group_rotation(g))
            Z_rot = Z[:, start:end] @ R.T
            sum_term = xp.sum(Z_rot**2, axis=1) / 4000
            prod_term = xp.prod(xp.cos(Z_rot / xp.sqrt(i_g)), axis=1)
            result = result + sum_term - prod_term + 1

        return result


# =============================================================================
# F14-F18: D/(2m)-group Rotated Functions (overlapping groups)
# =============================================================================


class OverlapSchwefel(CEC2010NonSeparableFunction):
    """F14: D/2m-group Shifted and m-rotated Schwefel Function."""

    _spec = {
        **CEC2010NonSeparableFunction._spec,
        "func_id": 14,
        "name": "OverlapSchwefel",
        "default_bounds": (-100.0, 100.0),
        "unimodal": True,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            result = 0.0
            # Use half the groups with double coverage
            for g in range(self.n_groups // 2):
                start = g * 2 * self.m
                end = start + self.m
                if end <= len(z):
                    R = self._get_group_rotation(g)
                    z_rot = R @ z[start:end]
                    result += _schwefel(z_rot)
            return result
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        m = self.m

        result = xp.zeros(X.shape[0])
        for g in range(self.n_groups // 2):
            start = g * 2 * m
            end = start + m
            if end <= X.shape[1]:
                R = xp.asarray(self._get_group_rotation(g))
                Z_rot = Z[:, start:end] @ R.T
                for i in range(m):
                    result = result + xp.sum(Z_rot[:, :i + 1], axis=1)**2

        return result


class OverlapRosenbrock(CEC2010NonSeparableFunction):
    """F15: D/2m-group Shifted and m-rotated Rosenbrock Function."""

    _spec = {
        **CEC2010NonSeparableFunction._spec,
        "func_id": 15,
        "name": "OverlapRosenbrock",
        "default_bounds": (-100.0, 100.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            result = 0.0
            for g in range(self.n_groups // 2):
                start = g * 2 * self.m
                end = start + self.m
                if end <= len(z):
                    R = self._get_group_rotation(g)
                    z_rot = R @ z[start:end]
                    result += _rosenbrock(z_rot)
            return result
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        m = self.m

        result = xp.zeros(X.shape[0])
        for g in range(self.n_groups // 2):
            start = g * 2 * m
            end = start + m
            if end <= X.shape[1]:
                R = xp.asarray(self._get_group_rotation(g))
                Z_rot = Z[:, start:end] @ R.T
                t1 = 100 * (Z_rot[:, 1:] - Z_rot[:, :-1]**2)**2
                t2 = (Z_rot[:, :-1] - 1)**2
                result = result + xp.sum(t1 + t2, axis=1)

        return result


class NonSepRastrigin(CEC2010NonSeparableFunction):
    """F16: D/2m-group Shifted Rastrigin Function."""

    _spec = {
        **CEC2010NonSeparableFunction._spec,
        "func_id": 16,
        "name": "NonSepRastrigin",
        "default_bounds": (-5.0, 5.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            return _rastrigin(z)
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        return xp.sum(Z**2 - 10 * xp.cos(2 * np.pi * Z) + 10, axis=1)


class NonSepAckley(CEC2010NonSeparableFunction):
    """F17: D/2m-group Shifted Ackley Function."""

    _spec = {
        **CEC2010NonSeparableFunction._spec,
        "func_id": 17,
        "name": "NonSepAckley",
        "default_bounds": (-32.0, 32.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            return _ackley(z)
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        D = Z.shape[1]
        term1 = -20 * xp.exp(-0.2 * xp.sqrt(xp.sum(Z**2, axis=1) / D))
        term2 = -xp.exp(xp.sum(xp.cos(2 * np.pi * Z), axis=1) / D)
        return term1 + term2 + 20 + np.e


class NonSepGriewank(CEC2010NonSeparableFunction):
    """F18: D/2m-group Shifted Griewank Function."""

    _spec = {
        **CEC2010NonSeparableFunction._spec,
        "func_id": 18,
        "name": "NonSepGriewank",
        "default_bounds": (-600.0, 600.0),
        "unimodal": False,
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._permute(self._shift(x, self.func_id))
            return _griewank(z)
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        perm = self._get_permutation()
        Z = Z[:, perm]
        D = Z.shape[1]
        i = xp.arange(1, D + 1, dtype=X.dtype)
        sum_term = xp.sum(Z**2, axis=1) / 4000
        prod_term = xp.prod(xp.cos(Z / xp.sqrt(i)), axis=1)
        return sum_term - prod_term + 1


# =============================================================================
# F19-F20: Composition Functions
# =============================================================================


class Composition1(CEC2010CompositionFunction):
    """F19: Schwefel's Composition Function 1."""

    _spec = {
        **CEC2010CompositionFunction._spec,
        "func_id": 19,
        "name": "Composition1",
        "default_bounds": (-5.0, 5.0),
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            # Composition of sphere, ackley, rastrigin
            f1 = np.sum(z**2)
            f2 = _ackley(z)
            f3 = _rastrigin(z)
            # Weighted combination
            return f1 + f2 + f3
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        D = Z.shape[1]

        # Sphere
        f1 = xp.sum(Z**2, axis=1)

        # Ackley
        t1 = -20 * xp.exp(-0.2 * xp.sqrt(xp.sum(Z**2, axis=1) / D))
        t2 = -xp.exp(xp.sum(xp.cos(2 * np.pi * Z), axis=1) / D)
        f2 = t1 + t2 + 20 + np.e

        # Rastrigin
        f3 = xp.sum(Z**2 - 10 * xp.cos(2 * np.pi * Z) + 10, axis=1)

        return f1 + f2 + f3


class Composition2(CEC2010CompositionFunction):
    """F20: Schwefel's Composition Function 2."""

    _spec = {
        **CEC2010CompositionFunction._spec,
        "func_id": 20,
        "name": "Composition2",
        "default_bounds": (-5.0, 5.0),
    }

    def _create_objective_function(self) -> None:
        def objective(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift(x, self.func_id)
            # Composition with different weights
            f1 = np.sum(z**2)
            f2 = _griewank(z)
            f3 = _rastrigin(z)
            return 0.5 * f1 + 0.3 * f2 + 0.2 * f3
        self.pure_objective_function = objective

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.func_id)
        D = Z.shape[1]

        # Sphere
        f1 = xp.sum(Z**2, axis=1)

        # Griewank
        i = xp.arange(1, D + 1, dtype=X.dtype)
        f2 = xp.sum(Z**2, axis=1) / 4000 - xp.prod(xp.cos(Z / xp.sqrt(i)), axis=1) + 1

        # Rastrigin
        f3 = xp.sum(Z**2 - 10 * xp.cos(2 * np.pi * Z) + 10, axis=1)

        return 0.5 * f1 + 0.3 * f2 + 0.2 * f3


# =============================================================================
# All CEC 2010 functions
# =============================================================================

CEC2010_ALL = [
    SeparableElliptic,
    SeparableRastrigin,
    SeparableAckley,
    SingleGroupElliptic,
    SingleGroupRastrigin,
    SingleGroupAckley,
    SingleGroupSchwefel,
    MultiGroupElliptic,
    MultiGroupRastrigin,
    MultiGroupAckley,
    MultiGroupSchwefel,
    MultiGroupRosenbrock,
    MultiGroupGriewank,
    OverlapSchwefel,
    OverlapRosenbrock,
    NonSepRastrigin,
    NonSepAckley,
    NonSepGriewank,
    Composition1,
    Composition2,
]

CEC2010_SEPARABLE = [
    SeparableElliptic,
    SeparableRastrigin,
    SeparableAckley,
]

CEC2010_PARTIAL_SEPARABLE = [
    SingleGroupElliptic,
    SingleGroupRastrigin,
    SingleGroupAckley,
    SingleGroupSchwefel,
    MultiGroupElliptic,
    MultiGroupRastrigin,
    MultiGroupAckley,
    MultiGroupSchwefel,
    MultiGroupRosenbrock,
    MultiGroupGriewank,
]

CEC2010_NONSEPARABLE = [
    OverlapSchwefel,
    OverlapRosenbrock,
    NonSepRastrigin,
    NonSepAckley,
    NonSepGriewank,
]

CEC2010_COMPOSITION = [
    Composition1,
    Composition2,
]
