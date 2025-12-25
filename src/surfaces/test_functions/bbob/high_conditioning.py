# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""BBOB Functions with High Conditioning and Unimodal (f10-f14)."""

from typing import Any, Dict

import numpy as np

from ._base_bbob import BBOBFunction


class EllipsoidalRotated(BBOBFunction):
    """f10: Ellipsoidal Function (Rotated).

    Rotated version of the separable ellipsoidal function.
    Tests optimization on ill-conditioned, non-separable problems.

    Properties:
    - Unimodal
    - Non-separable
    - Ill-conditioned (condition number: 10^6)
    """

    _spec = {
        "name": "Ellipsoidal Function",
        "func_id": 10,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        i = np.arange(self.n_dim)
        coeffs = np.power(1e6, i / (self.n_dim - 1)) if self.n_dim > 1 else np.ones(1)

        def ellipsoidal(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self.t_osz(self.R @ (x - self.x_opt))
            return np.sum(coeffs * z**2) + self.f_opt

        self.pure_objective_function = ellipsoidal


class Discus(BBOBFunction):
    """f11: Discus Function.

    One coordinate is highly sensitive, all others are flat.
    Condition number: 10^6.

    Properties:
    - Unimodal
    - Non-separable
    - Ill-conditioned
    """

    _spec = {
        "name": "Discus Function",
        "func_id": 11,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def discus(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self.t_osz(self.R @ (x - self.x_opt))
            return 1e6 * z[0] ** 2 + np.sum(z[1:] ** 2) + self.f_opt

        self.pure_objective_function = discus


class BentCigar(BBOBFunction):
    """f12: Bent Cigar Function.

    One sensitive direction, perpendicular directions are flat.
    Condition number: 10^6.

    Properties:
    - Unimodal
    - Non-separable
    - Ill-conditioned
    """

    _spec = {
        "name": "Bent Cigar Function",
        "func_id": 12,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def bent_cigar(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self.R @ self.t_asy(self.R @ (x - self.x_opt), 0.5)
            return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2) + self.f_opt

        self.pure_objective_function = bent_cigar


class SharpRidge(BBOBFunction):
    """f13: Sharp Ridge Function.

    Ridge along one axis, sharp perpendicular to it.
    Tests algorithm behavior on ridges.

    Properties:
    - Unimodal
    - Non-separable
    - Ridge structure
    """

    _spec = {
        "name": "Sharp Ridge Function",
        "func_id": 13,
        "unimodal": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        Lambda = self.lambda_alpha(10)

        def sharp_ridge(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self.Q @ Lambda @ self.R @ (x - self.x_opt)
            return z[0] ** 2 + 100 * np.sqrt(np.sum(z[1:] ** 2)) + self.f_opt

        self.pure_objective_function = sharp_ridge


class DifferentPowers(BBOBFunction):
    """f14: Different Powers Function.

    Sum of different powers, from quadratic to high degree.
    Sensitivity increases with index.

    Properties:
    - Unimodal
    - Non-separable
    - Varying local sensitivity
    """

    _spec = {
        "name": "Different Powers Function",
        "func_id": 14,
        "unimodal": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        i = np.arange(self.n_dim)
        exponents = 2 + 4 * i / (self.n_dim - 1) if self.n_dim > 1 else 2 * np.ones(1)

        def different_powers(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self.R @ (x - self.x_opt)
            return np.sqrt(np.sum(np.abs(z) ** exponents)) + self.f_opt

        self.pure_objective_function = different_powers
