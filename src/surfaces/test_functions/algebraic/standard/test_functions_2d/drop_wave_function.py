# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class DropWaveFunction(AlgebraicFunction):
    """Drop-Wave two-dimensional test function.

    A highly multimodal function with many local minima arranged in a
    concentric wave pattern.

    The function is defined as:

    .. math::

        f(x, y) = -\\frac{1 + \\cos(12\\sqrt{x^2 + y^2})}{0.5(x^2 + y^2) + 2}

    The global minimum is :math:`f(0, 0) = -1`.

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
    >>> from surfaces.test_functions import DropWaveFunction
    >>> func = DropWaveFunction()
    >>> result = func({"x0": 0.0, "x1": 0.0})
    >>> abs(result + 1.0) < 1e-10
    True
    """

    name = "Drop Wave Function"
    _name_ = "drop_wave_function"
    __name__ = "DropWaveFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = -1.0
    x_global = (0.0, 0.0)

    default_bounds = (-5.0, 5.0)
    n_dim = 2

    latex_formula = r"f(x, y) = -\frac{1 + \cos\left(12\sqrt{x^2 + y^2}\right)}{0.5(x^2 + y^2) + 2}"
    pgfmath_formula = "-(1 + cos(deg(12*sqrt(#1^2 + #2^2))))/(0.5*(#1^2 + #2^2) + 2)"

    # Function sheet attributes
    tagline = (
        "Concentric circular waves radiating from the origin. "
        "Many local minima in a ripple pattern challenge local search."
    )
    display_bounds = (-5.0, 5.0)
    reference = None
    reference_url = "https://www.sfu.ca/~ssurjano/drop.html"

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
        def drop_wave_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            return -(1 + math.cos(12 * math.sqrt(x**2 + y**2))) / (0.5 * (x**2 + y**2) + 2)

        self.pure_objective_function = drop_wave_function

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

        return -(1 + xp.cos(12 * xp.sqrt(x**2 + y**2))) / (0.5 * (x**2 + y**2) + 2)

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
