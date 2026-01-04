# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class SineProductFunction(AlgebraicFunction):
    """Sine Product one-dimensional test function.

    A one-dimensional test function that multiplies a linear term
    with a sine function. The amplitude of oscillations grows with x,
    creating increasingly deep valleys at larger values.

    The function is defined as:

    .. math::

        f(x) = -x \\sin(x)

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
        Default parameter bounds (0, 10).

    References
    ----------
    .. [1] AMPGO (Adaptive Memory Programming for Global Optimization)
       benchmark suite, Problem10.
       http://infinity77.net/global_optimization/test_functions_1d.html

    .. [2] Gavana, A. (2013). "Global Optimization Benchmarks and AMPGO".

    Examples
    --------
    >>> from surfaces.test_functions import SineProductFunction
    >>> func = SineProductFunction()
    >>> func({"x0": 7.9787})  # Near global minimum
    -7.916...
    >>> search_space = func.search_space
    >>> len(search_space)
    1
    """

    name = "Sine Product Function"
    _name_ = "sine_product_function"
    __name__ = "SineProductFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": True,
        "scalable": False,
    }

    f_global = -7.916727371587256
    x_global = np.array([7.9786653537049483])

    default_bounds = (0.0, 10.0)
    n_dim = 1

    latex_formula = r"f(x) = -x \sin(x)"
    pgfmath_formula = "-#1 * sin(deg(#1))"

    # Function sheet attributes
    tagline = (
        "Growing oscillations with increasing amplitude. The deepest valleys "
        "appear at larger x values where the linear term dominates."
    )
    display_bounds = (0.0, 10.0)
    reference = "Gavana (2013)"
    reference_url = "http://infinity77.net/global_optimization/test_functions_1d.html"

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
        def sine_product_function(params: Dict[str, Any]) -> float:
            x = params["x0"]

            return -x * np.sin(x)

        self.pure_objective_function = sine_product_function

    def _search_space(
        self,
        min: float = 0.0,
        max: float = 10.0,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
