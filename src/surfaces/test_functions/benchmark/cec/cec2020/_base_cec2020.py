# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for CEC 2020 benchmark functions."""

from surfaces._array_utils import ArrayLike

from .._base_cec import CECFunction


class CEC2020Function(CECFunction):
    """Base class for CEC 2020 benchmark functions.

    CEC 2020 functions are shifted and/or rotated versions of classical
    optimization test functions. Each function has:
    - A function ID (1-10)
    - A specific f_bias value (global optimum)
    - Search bounds of [-100, 100]^D
    - Support for dimensions: 5, 10, 15, 20

    Function categories:
    - F1: Unimodal (Bent Cigar)
    - F2-F4: Basic multimodal (Schwefel, Lunacek bi-Rastrigin, Expanded Griewank-Rosenbrock)
    - F5-F7: Hybrid functions
    - F8-F10: Composition functions

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 5, 10, 15, 20.
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Attributes
    ----------
    func_id : int
        Function ID (1-10), set by subclass.
    f_global : float
        Global optimum value (f_bias).

    References
    ----------
    Yue, C. T., Price, K. V., Suganthan, P. N., Liang, J. J., Ali, M. Z.,
    Qu, B. Y., Awad, N. H., & Biswas, P. P. (2019). Problem definitions and
    evaluation criteria for the CEC 2020 special session and competition on
    single objective bound constrained numerical optimization.
    """

    data_prefix = "cec2020"
    supported_dims = (5, 10, 15, 20)

    # f_bias values for each function (defines f_global)
    _f_bias = {
        1: 100.0,
        2: 1100.0,
        3: 700.0,
        4: 1900.0,
        5: 1700.0,
        6: 1600.0,
        7: 2100.0,
        8: 2200.0,
        9: 2400.0,
        10: 2500.0,
    }

    @property
    def f_global(self) -> float:
        """Global optimum value for this function.

        CEC 2020: f* = f_bias[func_id]
        """
        return self._f_bias.get(self.func_id, 0.0)

    def _shift_rotate(self, x):
        """Apply shift then rotation: z = M @ (x - o)."""
        return self._rotate(self._shift(x))

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
        return self._batch_rotate(self._batch_shift(X))
