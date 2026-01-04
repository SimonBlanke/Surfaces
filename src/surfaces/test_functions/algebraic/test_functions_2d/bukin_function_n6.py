# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class BukinFunctionN6(AlgebraicFunction):
    """Bukin N.6 two-dimensional test function.

    A non-convex function with a ridge along a parabola, making it challenging
    for many optimization algorithms.

    The function is defined as:

    .. math::

        f(x, y) = 100 \\sqrt{|y - 0.01x^2|} + 0.01|x + 10|

    The global minimum is :math:`f(-10, 1) = 0`.

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
    default_bounds : tuple
        Default parameter bounds (-8.0, 8.0).

    Examples
    --------
    >>> from surfaces.test_functions import BukinFunctionN6
    >>> func = BukinFunctionN6()
    >>> result = func({"x0": -10.0, "x1": 1.0})
    >>> abs(result) < 1e-10
    True
    """

    name = "Bukin Function N6"
    _name_ = "bukin_function_n6"
    __name__ = "BukinFunctionN6"

    _spec = {
        "convex": False,
        "unimodal": True,
        "separable": False,
        "scalable": False,
        "differentiable": False,
    }

    f_global = 0.0
    x_global = np.array([-10.0, 1.0])

    default_bounds = (-8.0, 8.0)
    n_dim = 2

    latex_formula = r"f(x, y) = 100\sqrt{|y - 0.01x^2|} + 0.01|x + 10|"
    pgfmath_formula = "100*sqrt(abs(#2 - 0.01*#1^2)) + 0.01*abs(#1 + 10)"

    # Function sheet attributes
    tagline = (
        "A narrow ridge along a parabola with the minimum in a steep drop-off. "
        "Extremely challenging for gradient-based methods."
    )
    display_bounds = (-15.0, 5.0)
    reference = None
    reference_url = "https://www.sfu.ca/~ssurjano/bukin6.html"

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
        def bukin_function_n6(params):
            x = params["x0"]
            y = params["x1"]

            return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

        self.pure_objective_function = bukin_function_n6

    def _search_space(
        self,
        min: float = -8,
        max: float = 8,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
