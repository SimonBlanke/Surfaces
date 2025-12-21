# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for CEC 2013 benchmark functions."""

from typing import Optional

import numpy as np

from .._base_cec import CECFunction


class CEC2013Function(CECFunction):
    """Base class for CEC 2013 benchmark functions.

    CEC 2013 functions are shifted and/or rotated versions of classical
    optimization test functions. Each function has:
    - A function ID (1-28)
    - A global optimum value f* = -1400 + (func_id - 1) * 100
    - Search bounds of [-100, 100]^D
    - Support for dimensions: 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    func_id : int
        Function ID (1-28), set by subclass.
    shift_index : int
        Which shift vector to use (1-10 available in data files).
    uses_rotation : bool
        Whether to apply rotation in shift_rotate transformation.
    f_global : float
        Global optimum value.

    References
    ----------
    Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernandez-Diaz, A. G. (2013).
    Problem definitions and evaluation criteria for the CEC 2013 special
    session on real-parameter optimization.
    """

    data_prefix = "cec2013"
    shift_index: int = 1
    uses_rotation: bool = True
    supported_dims = (2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

    @property
    def f_global(self) -> float:
        """Global optimum value for this function.

        CEC 2013 formula: f* = -1400 + (func_id - 1) * 100
        """
        return float(-1400 + (self.func_id - 1) * 100)

    @property
    def x_global(self) -> Optional[np.ndarray]:
        """Global optimum location (the shift vector)."""
        return self._get_shift_vector(self.shift_index)

    def _get_shift_vector(self, index: int = None) -> np.ndarray:
        """Get the shift vector for this function.

        CEC 2013 uses shift_index instead of func_id for shift vectors.
        """
        if index is None:
            index = self.shift_index
        return super()._get_shift_vector(index)

    def _get_rotation_matrix(self, index: int = None) -> np.ndarray:
        """Get a rotation matrix for this function.

        CEC 2013 uses shift_index instead of func_id for rotation matrices.
        """
        if index is None:
            index = self.shift_index
        return super()._get_rotation_matrix(index)

    def _shift_rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply shift then rotation: z = M @ (x - o).

        CEC 2013 functions may optionally skip rotation based on uses_rotation.
        """
        shifted = self._shift(x, self.shift_index)
        if self.uses_rotation:
            return self._rotate(shifted, self.shift_index)
        return shifted
