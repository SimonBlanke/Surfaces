# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class QuadraticExponentialFunction(AlgebraicFunction):
    """Quadratic-Exponential one-dimensional test function.

    A one-dimensional test function combining a quadratic polynomial
    with an exponential decay term. The function has a smooth,
    single-valley landscape.

    The function is defined as:

    .. math::

        f(x) = -(16x^2 - 24x + 5) e^{-x}

    Parameters
    ----------
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Attributes
    ----------
    n_dim : int
        Number of dimensions (always 1).
    default_bounds : tuple
        Default parameter bounds (1.9, 3.9).

    References
    ----------
    .. [1] AMPGO (Adaptive Memory Programming for Global Optimization)
       benchmark suite, Problem04.
       http://infinity77.net/global_optimization/test_functions_1d.html

    .. [2] Gavana, A. (2013). "Global Optimization Benchmarks and AMPGO".

    Examples
    --------
    >>> from surfaces.test_functions import QuadraticExponentialFunction
    >>> func = QuadraticExponentialFunction()
    >>> func({"x0": 2.868})  # Near global minimum
    -3.850...
    >>> search_space = func.search_space
    >>> len(search_space)
    1
    """

    name = "Quadratic Exponential Function"
    _name_ = "quadratic_exponential_function"
    __name__ = "QuadraticExponentialFunction"

    _spec = {
        "convex": False,
        "unimodal": True,
        "separable": True,
        "scalable": False,
    }

    f_global = -3.8504507087979953
    x_global = (2.8680325095605212,)

    default_bounds = (1.9, 3.9)
    n_dim = 1

    latex_formula = r"f(x) = -(16x^2 - 24x + 5) e^{-x}"
    pgfmath_formula = "-(16*#1^2 - 24*#1 + 5) * exp(-#1)"

    # Function sheet attributes
    tagline = (
        "A smooth single-valley landscape. The exponential decay moderates "
        "the quadratic growth, creating a well-defined minimum."
    )
    display_bounds = (1.9, 3.9)
    reference = "Gavana (2013)"
    reference_url = "http://infinity77.net/global_optimization/test_functions_1d.html"

    def __init__(
        self,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)
        self.n_dim = 1

    def _create_objective_function(self) -> None:
        def quadratic_exponential_function(params: Dict[str, Any]) -> float:
            x = params["x0"]

            return -(16 * x**2 - 24 * x + 5) * math.exp(-x)

        self.pure_objective_function = quadratic_exponential_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation.

        Parameters
        ----------
        X : ArrayLike
            Array of shape (n_points, 1).

        Returns
        -------
        ArrayLike
            Array of shape (n_points,).
        """
        xp = get_array_namespace(X)

        x = X[:, 0]
        return -(16 * x**2 - 24 * x + 5) * xp.exp(-x)

    def _search_space(
        self,
        min: float = 1.9,
        max: float = 3.9,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
