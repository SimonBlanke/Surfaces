# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class GoldsteinPriceFunction(AlgebraicFunction):
    """Goldstein-Price two-dimensional test function.

    A polynomial function with several local minima, commonly used in
    benchmarking optimization algorithms.

    The global minimum is :math:`f(0, -1) = 3`.

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
        Default parameter bounds (-2.0, 2.0).

    Examples
    --------
    >>> from surfaces.test_functions import GoldsteinPriceFunction
    >>> func = GoldsteinPriceFunction()
    >>> result = func({"x0": 0.0, "x1": -1.0})
    >>> abs(result - 3.0) < 1e-10
    True
    """

    name = "Goldstein Price Function"
    _name_ = "goldstein_price_function"
    __name__ = "GoldsteinPriceFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = 3.0
    x_global = (0.0, -1.0)

    default_bounds = (-2.0, 2.0)
    n_dim = 2

    latex_formula = r"f(x, y) = \left[1 + (x + y + 1)^2(19 - 14x + 3x^2 - 14y + 6xy + 3y^2)\right]\left[30 + (2x - 3y)^2(18 - 32x + 12x^2 + 48y - 36xy + 27y^2)\right]"
    pgfmath_formula = "(1 + (#1 + #2 + 1)^2*(19 - 14*#1 + 3*#1^2 - 14*#2 + 6*#1*#2 + 3*#2^2))*(30 + (2*#1 - 3*#2)^2*(18 - 32*#1 + 12*#1^2 + 48*#2 - 36*#1*#2 + 27*#2^2))"

    # Function sheet attributes
    tagline = (
        "Multiple local minima with large value differences. "
        "The polynomial structure creates a complex multi-scale landscape."
    )
    display_bounds = (-2.0, 2.0)
    reference = "Goldstein & Price (1971)"
    reference_url = "https://www.sfu.ca/~ssurjano/goldpr.html"

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
        def goldstein_price_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            loss1 = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
            loss2 = 30 + (2 * x - 3 * y) ** 2 * (
                18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2
            )

            return loss1 * loss2

        self.pure_objective_function = goldstein_price_function

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
        x = X[:, 0]
        y = X[:, 1]

        loss1 = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
        loss2 = 30 + (2 * x - 3 * y) ** 2 * (
            18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2
        )

        return loss1 * loss2

    def _search_space(
        self,
        min: float = -2,
        max: float = 2,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
