# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2014 Unimodal Functions (F1-F3).

These functions have a single global optimum and no local optima.
"""

from typing import Any, Dict

import numpy as np

from ._base_cec2014 import CEC2014Function


class RotatedHighConditionedElliptic(CEC2014Function):
    """F1: Rotated High Conditioned Elliptic Function.

    A unimodal function with high condition number, making it difficult
    for optimization algorithms that don't handle ill-conditioning well.

    Properties:
    - Unimodal
    - Non-separable (due to rotation)
    - Scalable
    - Condition number: 10^6

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds.
    """

    _spec = {
        "name": "Rotated High Conditioned Elliptic Function",
        "func_id": 1,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def elliptic(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            D = self.n_dim
            result = 0.0
            for i in range(D):
                result += (10**6) ** (i / (D - 1)) * z[i] ** 2

            return result + self.f_global

        self.pure_objective_function = elliptic


class RotatedBentCigar(CEC2014Function):
    """F2: Rotated Bent Cigar Function.

    A unimodal function with a narrow ridge. The condition number is 10^6.

    Properties:
    - Unimodal
    - Non-separable (due to rotation)
    - Scalable
    - Has a narrow ridge

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds.
    """

    _spec = {
        "name": "Rotated Bent Cigar Function",
        "func_id": 2,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def bent_cigar(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            result = z[0] ** 2 + 10**6 * np.sum(z[1:] ** 2)

            return result + self.f_global

        self.pure_objective_function = bent_cigar


class RotatedDiscus(CEC2014Function):
    """F3: Rotated Discus Function.

    A unimodal function where one variable has much larger contribution
    than others. Also known as the Rotated Tablet function.

    Properties:
    - Unimodal
    - Non-separable (due to rotation)
    - Scalable
    - Condition number: 10^6

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds.
    """

    _spec = {
        "name": "Rotated Discus Function",
        "func_id": 3,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def discus(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            result = 10**6 * z[0] ** 2 + np.sum(z[1:] ** 2)

            return result + self.f_global

        self.pure_objective_function = discus
