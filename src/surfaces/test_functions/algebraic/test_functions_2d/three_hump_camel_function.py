# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class ThreeHumpCamelFunction(AlgebraicFunction):
    """Three-Hump Camel two-dimensional test function.

    A function with three local minima, two of which are symmetric about
    the origin.

    The function is defined as:

    .. math::

        f(x, y) = 2x^2 - 1.05x^4 + \\frac{x^6}{6} + xy + y^2

    The global minimum is :math:`f(0, 0) = 0`.

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
        Default parameter bounds (-5.0, 5.0).

    Examples
    --------
    >>> from surfaces.test_functions import ThreeHumpCamelFunction
    >>> func = ThreeHumpCamelFunction()
    >>> result = func({"x0": 0.0, "x1": 0.0})
    >>> abs(result) < 1e-10
    True
    """

    name = "Three Hump Camel Function"
    _name_ = "three_hump_camel_function"
    __name__ = "ThreeHumpCamelFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = 0.0
    x_global = np.array([0.0, 0.0])

    default_bounds = (-5.0, 5.0)
    n_dim = 2

    latex_formula = r"f(x, y) = 2x^2 - 1.05x^4 + \frac{x^6}{6} + xy + y^2"
    pgfmath_formula = "2*#1^2 - 1.05*#1^4 + #1^6/6 + #1*#2 + #2^2"

    # Function sheet attributes
    tagline = (
        "Three humps with the global minimum at the center. "
        "The polynomial creates an asymmetric camel-back shape."
    )
    display_bounds = (-2.0, 2.0)
    reference = None
    reference_url = "https://www.sfu.ca/~ssurjano/camel3.html"

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
        def three_hump_camel_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            return 2 * x**2 - 1.05 * x**4 + x**6 / 6 + x * y + y**2

        self.pure_objective_function = three_hump_camel_function

    def _search_space(
        self,
        min: float = -5,
        max: float = 5,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
