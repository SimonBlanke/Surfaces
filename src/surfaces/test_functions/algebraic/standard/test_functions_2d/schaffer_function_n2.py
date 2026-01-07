# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class SchafferFunctionN2(AlgebraicFunction):
    """Schaffer N.2 two-dimensional test function.

    A multimodal function with many local minima.

    The function is defined as:

    .. math::

        f(x, y) = 0.5 + \\frac{\\sin^2(x^2 - y^2) - 0.5}
        {[1 + 0.001(x^2 + y^2)]^2}

    The global minimum is :math:`f(0, 0) = 0`.

    Parameters
    ----------
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Attributes
    ----------
    n_dim : int
        Number of dimensions (always 2).
    default_bounds : tuple
        Default parameter bounds (-50.0, 50.0).

    Examples
    --------
    >>> from surfaces.test_functions import SchafferFunctionN2
    >>> func = SchafferFunctionN2()
    >>> result = func({"x0": 0.0, "x1": 0.0})
    """

    name = "Schaffer Function N2"
    _name_ = "schaffer_function_n2"
    __name__ = "SchafferFunctionN2"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = 0.0
    x_global = (0.0, 0.0)

    default_bounds = (-50.0, 50.0)
    n_dim = 2

    latex_formula = r"f(x, y) = 0.5 + \frac{\sin^2(x^2 - y^2) - 0.5}{[1 + 0.001(x^2 + y^2)]^2}"
    pgfmath_formula = "0.5 + (sin(deg(#1^2 - #2^2))^2 - 0.5)/((1 + 0.001*(#1^2 + #2^2))^2)"

    # Function sheet attributes
    tagline = (
        "Concentric rings of local minima around the origin. "
        "The pattern slowly decays outward, testing long-range exploration."
    )
    display_bounds = (-10.0, 10.0)
    reference = "Schaffer (1994)"
    reference_url = "https://www.sfu.ca/~ssurjano/schaffer2.html"

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
        self.n_dim = 2

    def _create_objective_function(self) -> None:
        def schaffer_function_n2(params):
            x = params["x0"]
            y = params["x1"]

            return 0.5 + (math.sin(x**2 - y**2) ** 2 - 0.5) / ((1 + 0.001 * (x**2 + y**2)) ** 2)

        self.pure_objective_function = schaffer_function_n2

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation.

        Parameters
        ----------
        X : ArrayLike
            Array of shape (n_points, 2).

        Returns
        -------
        ArrayLike
            Array of shape (n_points,).
        """
        xp = get_array_namespace(X)

        x = X[:, 0]
        y = X[:, 1]

        return 0.5 + (xp.sin(x**2 - y**2) ** 2 - 0.5) / ((1 + 0.001 * (x**2 + y**2)) ** 2)

    def _search_space(
        self,
        min: float = -50,
        max: float = 50,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
