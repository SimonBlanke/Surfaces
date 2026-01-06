# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for CEC 2014 benchmark functions."""

from .._base_cec import CECFunction


class CEC2014Function(CECFunction):
    """Base class for CEC 2014 benchmark functions.

    CEC 2014 functions are shifted and/or rotated versions of classical
    optimization test functions. Each function has:
    - A function ID (1-30)
    - A global optimum value f* = func_id * 100
    - Search bounds of [-100, 100]^D
    - Support for dimensions: 10, 20, 30, 50, 100

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    func_id : int
        Function ID (1-30), set by subclass.
    f_global : float
        Global optimum value (func_id * 100).

    References
    ----------
    Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013).
    Problem definitions and evaluation criteria for the CEC 2014 special
    session and competition on single objective real-parameter numerical
    optimization.
    """

    data_prefix = "cec2014"
    supported_dims = (10, 20, 30, 50, 100)

    @property
    def f_global(self) -> float:
        """Global optimum value for this function.

        CEC 2014 formula: f* = func_id * 100
        """
        return float(self.func_id * 100)

    def _shift_rotate(self, x):
        """Apply shift then rotation: z = M @ (x - o)."""
        return self._rotate(self._shift(x))
