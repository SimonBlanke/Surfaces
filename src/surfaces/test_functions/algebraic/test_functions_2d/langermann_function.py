# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class LangermannFunction(AlgebraicFunction):
    """Langermann two-dimensional test function.

    A multimodal function with many unevenly distributed local minima.

    Parameters
    ----------
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    n_dim : int
        Number of dimensions (always 2).
    c : ndarray
        Coefficient vector.
    m : int
        Number of terms in the summation.
    A : ndarray
        Matrix of center coordinates.

    Examples
    --------
    >>> from surfaces.test_functions import LangermannFunction
    >>> func = LangermannFunction()
    >>> result = func({"x0": 0.0, "x1": 0.0})
    """

    name = "Langermann Function"
    _name_ = "langermann_function"
    __name__ = "LangermannFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = None  # Complex to determine analytically
    x_global = None

    default_bounds = (-15.0, 15.0)
    n_dim = 2

    latex_formula = r"f(x, y) = \sum_{i=1}^{m} c_i \exp\left(-\frac{1}{\pi}\sum_{j=1}^{2}(x_j - A_{ji})^2\right) \cos\left(\pi\sum_{j=1}^{2}(x_j - A_{ji})^2\right)"
    pgfmath_formula = None  # Complex summation not expressible in pgfmath

    # Function sheet attributes
    tagline = (
        "Irregularly distributed local minima with varying depths. "
        "A challenging landscape without clear structure."
    )
    display_bounds = (0.0, 10.0)
    reference = None
    reference_url = "https://www.sfu.ca/~ssurjano/langer.html"

    c = np.array([1, 2, 5, 2, 3])
    m = 5
    A = np.array([[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]])

    def __init__(
        self,
        objective: str = "minimize",
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        noise: Optional["BaseNoise"] = None,
    ) -> None:
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)
        self.n_dim = 2

    def _create_objective_function(self) -> None:
        def langermann_function(params: Dict[str, Any]) -> float:
            loss_sum1 = 0

            for m in range(self.m):
                loss_sum1 += self.c[m]

                loss_sum2 = 0
                loss_sum3 = 0
                for dim in range(self.n_dim):
                    dim_str = "x" + str(dim)
                    x = params[dim_str]

                    loss_sum2 += x - self.A[dim, m]
                    loss_sum3 += x - self.A[dim, m]

                loss_sum2 *= -1 / np.pi
                loss_sum3 *= np.pi

            return loss_sum1 * np.exp(loss_sum2) * np.cos(loss_sum3)

        self.pure_objective_function = langermann_function

    def _search_space(
        self,
        min: float = -15,
        max: float = 15,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
