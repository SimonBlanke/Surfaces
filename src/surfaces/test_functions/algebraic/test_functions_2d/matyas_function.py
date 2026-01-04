# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class MatyasFunction(AlgebraicFunction):
    """Matyas two-dimensional test function.

    A bowl-shaped, unimodal function.

    The function is defined as:

    .. math::

        f(x, y) = 0.26(x^2 + y^2) - 0.48xy

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
        Default parameter bounds (-10.0, 10.0).

    Examples
    --------
    >>> from surfaces.test_functions import MatyasFunction
    >>> func = MatyasFunction()
    >>> result = func({"x0": 0.0, "x1": 0.0})
    >>> abs(result) < 1e-10
    True
    """

    name = "Matyas Function"
    _name_ = "matyas_function"
    __name__ = "MatyasFunction"

    _spec = {
        "convex": True,
        "unimodal": True,
        "separable": False,
        "scalable": False,
    }

    f_global = 0.0
    x_global = np.array([0.0, 0.0])

    default_bounds = (-10.0, 10.0)
    n_dim = 2

    latex_formula = r"f(x, y) = 0.26(x^2 + y^2) - 0.48xy"
    pgfmath_formula = "0.26*(#1^2 + #2^2) - 0.48*#1*#2"

    # Function sheet attributes
    tagline = (
        "A nearly flat, elongated bowl. "
        "Simple and convex, used as an easy baseline for testing optimization."
    )
    display_bounds = (-10.0, 10.0)
    reference = None
    reference_url = "https://www.sfu.ca/~ssurjano/matya.html"

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
        def matyas_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            return 0.26 * (x**2 + y**2) - 0.48 * x * y

        self.pure_objective_function = matyas_function

    def _search_space(
        self,
        min: float = -10,
        max: float = 10,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
