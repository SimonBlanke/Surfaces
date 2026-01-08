# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for CEC 2022 benchmark functions."""

from surfaces._array_utils import ArrayLike

from .._base_cec import CECFunction


class CEC2022Function(CECFunction):
    """Base class for CEC 2022 benchmark functions.

    CEC 2022 functions are shifted and/or rotated versions of classical
    optimization test functions. Each function has:
    - A function ID (1-12)
    - A specific f_bias value (global optimum)
    - Search bounds of [-100, 100]^D
    - Support for dimensions: 10, 20

    Function categories:
    - F1: Unimodal (Zakharov)
    - F2-F5: Basic multimodal (Rosenbrock, Schaffer F7, Rastrigin, Levy)
    - F6-F8: Hybrid functions
    - F9-F12: Composition functions

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20.
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Attributes
    ----------
    func_id : int
        Function ID (1-12), set by subclass.
    f_global : float
        Global optimum value (f_bias).

    References
    ----------
    Abhishek Kumar, Kenneth V. Price, Ali Wagdy Mohamed, Anas A. Hadi,
    P. N. Suganthan (2021). Problem definitions and evaluation criteria for
    the CEC 2022 special session and competition on single objective bound
    constrained numerical optimization.
    """

    data_prefix = "cec2022"
    supported_dims = (10, 20)

    # f_bias values for each function
    _f_bias = {
        1: 300.0,
        2: 400.0,
        3: 600.0,
        4: 800.0,
        5: 900.0,
        6: 1800.0,
        7: 2000.0,
        8: 2200.0,
        9: 2300.0,
        10: 2400.0,
        11: 2600.0,
        12: 2700.0,
    }

    @property
    def f_global(self) -> float:
        """Global optimum value for this function.

        CEC 2022: f* = f_bias[func_id]
        """
        return self._f_bias.get(self.func_id, 0.0)

    def _shift_rotate(self, x):
        """Apply shift then rotation: z = M @ (x - o)."""
        return self._rotate(self._shift(x))

    def _batch_shift_rotate(self, X: ArrayLike) -> ArrayLike:
        """Apply shift then rotation to batch: Z = (X - o) @ M.T."""
        return self._batch_rotate(self._batch_shift(X))
