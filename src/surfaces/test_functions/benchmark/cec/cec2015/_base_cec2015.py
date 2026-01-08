# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for CEC 2015 benchmark functions."""

from .._base_cec import CECFunction


class CEC2015Function(CECFunction):
    """Base class for CEC 2015 benchmark functions.

    CEC 2015 functions are shifted and/or rotated versions of classical
    optimization test functions. Each function has:
    - A function ID (1-15)
    - A global optimum value f* = func_id * 100
    - Search bounds of [-100, 100]^D
    - Support for dimensions: 10, 30

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 30.
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Attributes
    ----------
    func_id : int
        Function ID (1-15), set by subclass.
    f_global : float
        Global optimum value (func_id * 100).

    References
    ----------
    Liang, J. J., Qu, B. Y., Suganthan, P. N., & Chen, Q. (2014).
    Problem definitions and evaluation criteria for the CEC 2015
    competition on learning-based real-parameter single objective optimization.
    Technical Report, Zhengzhou University and Nanyang Technological University.
    """

    data_prefix = "cec2015"
    supported_dims = (10, 30)

    @property
    def f_global(self) -> float:
        """Global optimum value for this function.

        CEC 2015 formula: f* = func_id * 100
        """
        return float(self.func_id * 100)

    def _shift_rotate(self, x):
        """Apply shift then rotation: z = M @ (x - o)."""
        return self._rotate(self._shift(x))
