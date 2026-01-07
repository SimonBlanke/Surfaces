# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class McCormickFunction(AlgebraicFunction):
    """McCormick two-dimensional test function.

    A function with a single global minimum, commonly used for testing
    optimization algorithms.

    The function is defined as:

    .. math::

        f(x, y) = \\sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1

    The global minimum is :math:`f(-0.54719, -1.54719) = -1.9133`.

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
        Default parameter bounds (-5.0, 5.0).

    Examples
    --------
    >>> from surfaces.test_functions import McCormickFunction
    >>> func = McCormickFunction()
    >>> result = func({"x0": -0.54719, "x1": -1.54719})
    """

    name = "Mc Cormick Function"
    _name_ = "mccormick_function"
    __name__ = "McCormickFunction"

    _spec = {
        "convex": False,
        "unimodal": True,
        "separable": False,
        "scalable": False,
    }

    f_global = -1.9133
    x_global = (-0.54719, -1.54719)

    default_bounds = (-5.0, 5.0)
    n_dim = 2

    latex_formula = r"f(x, y) = \sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1"
    pgfmath_formula = "sin(deg(#1 + #2)) + (#1 - #2)^2 - 1.5*#1 + 2.5*#2 + 1"

    # Function sheet attributes
    tagline = (
        "A smooth landscape with a sinusoidal ridge. "
        "The single minimum lies in a gently curving valley."
    )
    display_bounds = (-3.0, 4.0)
    reference = None
    reference_url = "https://www.sfu.ca/~ssurjano/mccorm.html"

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
        def mccormick_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            return math.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

        self.pure_objective_function = mccormick_function

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

        return xp.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

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
