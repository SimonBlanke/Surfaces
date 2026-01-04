# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class GramacyAndLeeFunction(AlgebraicFunction):
    """Gramacy and Lee one-dimensional test function.

    A simple one-dimensional test function commonly used for testing
    optimization algorithms and surrogate modeling techniques.

    The function is defined as:

    .. math::

        f(x) = \\frac{\\sin(10\\pi x)}{2x} + (x - 1)^4

    Parameters
    ----------
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    n_dim : int
        Number of dimensions (always 1).
    default_bounds : tuple
        Default parameter bounds (0.5, 2.5).

    References
    ----------
    .. [1] Gramacy, R. B., & Lee, H. K. (2012). "Cases for the nugget in
       modeling computer experiments". Statistics and Computing, 22(3),
       713-722.

    Examples
    --------
    >>> from surfaces.test_functions import GramacyAndLeeFunction
    >>> func = GramacyAndLeeFunction()
    >>> func({"x0": 1.0})
    0.0
    >>> search_space = func.search_space
    >>> len(search_space)
    1
    """

    name = "Gramacy And Lee Function"
    _name_ = "gramacy_and_lee_function"
    __name__ = "GramacyAndLeeFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": True,
        "scalable": False,
    }

    f_global = -0.869011134989500
    x_global = np.array([0.548563444114526])

    default_bounds = (0.5, 2.5)
    n_dim = 1

    latex_formula = r"f(x) = \frac{\sin(10\pi x)}{2x} + (x - 1)^4"
    pgfmath_formula = "sin(deg(10*pi*#1))/(2*#1) + (#1 - 1)^4"

    # Function sheet attributes
    tagline = (
        "A compact oscillatory function with rapid frequency changes. "
        "Popular for testing Gaussian process surrogate models."
    )
    display_bounds = (0.5, 2.5)
    reference = "Gramacy & Lee (2012)"
    reference_url = "https://doi.org/10.1007/s11222-011-9275-5"

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
        self.n_dim = 1

    def _create_objective_function(self) -> None:
        def gramacy_and_lee_function(params: Dict[str, Any]) -> float:
            x = params["x0"]

            return (np.sin(10 * np.pi * x) / (2 * x)) + (x - 1) ** 4

        self.pure_objective_function = gramacy_and_lee_function

    def _search_space(
        self,
        min: float = 0.5,
        max: float = 2.5,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
