# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for CEC 2005 benchmark functions."""

from typing import Optional, Tuple

import numpy as np

from surfaces._array_utils import ArrayLike

from .._base_cec import CECFunction


class CEC2005Function(CECFunction):
    """Base class for CEC 2005 benchmark functions.

    CEC 2005 functions are shifted and/or rotated versions of classical
    optimization test functions. Each function has:
    - A function ID (1-25)
    - A fixed bias value (f_bias) as global optimum
    - Search bounds that vary by function type
    - Support for dimensions: 2, 10, 30, 50 (for rotated functions)

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. For rotated functions: 2, 10, 30, 50.
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Attributes
    ----------
    func_id : int
        Function ID (1-25), set by subclass.
    f_global : float
        Global optimum value (bias).

    References
    ----------
    Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y.-P.,
    Auger, A., & Tiwari, S. (2005). Problem definitions and evaluation
    criteria for the CEC 2005 special session on real-parameter optimization.
    Technical Report, Nanyang Technological University, Singapore.
    """

    data_prefix = "cec2005"

    # Rotated functions require these specific dimensions
    supported_dims: Tuple[int, ...] = (2, 10, 30, 50)

    # Non-rotated functions can use arbitrary dimensions
    # (subclasses can override supported_dims)

    # CEC2005 bias values for each function (fixed)
    _bias_values = {
        1: -450.0,   # Sphere
        2: -450.0,   # Schwefel 1.2
        3: -450.0,   # Rotated Elliptic
        4: -450.0,   # Schwefel 1.2 with Noise
        5: -310.0,   # Schwefel 2.6
        6: 390.0,    # Rosenbrock
        7: -180.0,   # Rotated Griewank
        8: -140.0,   # Rotated Ackley
        9: -330.0,   # Shifted Rastrigin
        10: -330.0,  # Rotated Rastrigin
        11: 90.0,    # Rotated Weierstrass
        12: -460.0,  # Schwefel 2.13
        13: -130.0,  # Expanded Griewank-Rosenbrock
        14: -300.0,  # Expanded Scaffer F6
        15: 120.0,   # Composition F1
        16: 120.0,   # Composition F2
        17: 120.0,   # Composition F3
        18: 10.0,    # Composition F4
        19: 10.0,    # Composition F5
        20: 10.0,    # Composition F6
        21: 360.0,   # Composition F7
        22: 360.0,   # Composition F8
        23: 360.0,   # Composition F9
        24: 260.0,   # Composition F10
        25: 260.0,   # Composition F11
    }

    @property
    def f_global(self) -> float:
        """Global optimum value for this function.

        CEC 2005 uses fixed bias values per function, unlike later years
        which use formulas like func_id * 100.
        """
        return self._bias_values[self.func_id]

    @property
    def x_global(self) -> Optional[np.ndarray]:
        """Global optimum location (the shift vector)."""
        return self._get_shift_vector(self.func_id)

    def _shift_rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply shift then rotation: z = M @ (x - o).

        Parameters
        ----------
        x : np.ndarray
            Input vector.

        Returns
        -------
        np.ndarray
            Shifted and rotated vector.
        """
        shifted = self._shift(x, self.func_id)
        return self._rotate(shifted, self.func_id)

    def _batch_shift_rotate(self, X: ArrayLike) -> ArrayLike:
        """Apply shift then rotation to batch: Z = (X - o) @ M.T.

        Parameters
        ----------
        X : ArrayLike
            Input batch of shape (n_points, n_dim).

        Returns
        -------
        ArrayLike
            Transformed batch of shape (n_points, n_dim).
        """
        shifted = self._batch_shift(X, self.func_id)
        return self._batch_rotate(shifted, self.func_id)


class CEC2005NonRotatedFunction(CEC2005Function):
    """Base class for CEC 2005 functions that don't require rotation.

    These functions are mathematically valid for arbitrary dimensions,
    but data files are only available for 2, 10, 30, 50.
    """

    # Same dimensions as rotated functions (data files only available for these)
    supported_dims = (2, 10, 30, 50)

    def _get_rotation_matrix(self, index: int = None) -> np.ndarray:
        """Return identity matrix (no rotation)."""
        return np.eye(self.n_dim)

    def _shift_rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply shift only (no rotation)."""
        return self._shift(x, self.func_id)

    def _batch_shift_rotate(self, X: ArrayLike) -> ArrayLike:
        """Apply shift only to batch (no rotation)."""
        return self._batch_shift(X, self.func_id)
