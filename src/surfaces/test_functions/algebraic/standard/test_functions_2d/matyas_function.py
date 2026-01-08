# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


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
    x_global = (0.0, 0.0)

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
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)
        self.n_dim = 2

    def _create_objective_function(self) -> None:
        def matyas_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            return 0.26 * (x**2 + y**2) - 0.48 * x * y

        self.pure_objective_function = matyas_function

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

        return 0.26 * (x**2 + y**2) - 0.48 * x * y

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
