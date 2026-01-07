# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class LeviFunctionN13(AlgebraicFunction):
    """Levi N.13 two-dimensional test function.

    A multimodal function with a single global minimum.

    The function is defined as:

    .. math::

        f(x, y) = \\sin^2(3\\pi x) + (x-1)^2(1 + \\sin^2(3\\pi y))
        + (y-1)^2(1 + \\sin^2(2\\pi y))

    The global minimum is :math:`f(1, 1) = 0`.

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
        Default parameter bounds (-10.0, 10.0).

    Examples
    --------
    >>> from surfaces.test_functions import LeviFunctionN13
    >>> func = LeviFunctionN13()
    >>> result = func({"x0": 1.0, "x1": 1.0})
    """

    name = "Levi Function N13"
    _name_ = "levi_function_n13"
    __name__ = "LeviFunctionN13"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = 0.0
    x_global = (1.0, 1.0)

    default_bounds = (-10.0, 10.0)
    n_dim = 2

    latex_formula = (
        r"f(x, y) = \sin^2(3\pi x) + (x - 1)^2(1 + \sin^2(3\pi y)) + (y - 1)^2(1 + \sin^2(2\pi y))"
    )
    pgfmath_formula = "sin(deg(3*pi*#1))^2 + (#1 - 1)^2*(1 + sin(deg(3*pi*#2))^2) + (#2 - 1)^2*(1 + sin(deg(2*pi*#2))^2)"

    # Function sheet attributes
    tagline = (
        "Sinusoidal modulation creates a grid of local minima. "
        "The minimum lies at a lattice point in this structured landscape."
    )
    display_bounds = (-10.0, 10.0)
    reference = None
    reference_url = "https://www.sfu.ca/~ssurjano/levy13.html"

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
        def levi_function_n13(params):
            x = params["x0"]
            y = params["x1"]

            return (
                math.sin(3 * math.pi * x) ** 2
                + (x + 1) ** 2 * (1 + math.sin(3 * math.pi * y) ** 2)
                + (y - 1) ** 2 * (1 + math.sin(3 * math.pi * y) ** 2)
            )

        self.pure_objective_function = levi_function_n13

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

        # Match the sequential implementation exactly
        return (
            xp.sin(3 * math.pi * x) ** 2
            + (x + 1) ** 2 * (1 + xp.sin(3 * math.pi * y) ** 2)
            + (y - 1) ** 2 * (1 + xp.sin(3 * math.pi * y) ** 2)
        )

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
