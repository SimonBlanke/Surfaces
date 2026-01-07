# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Dict, List, Optional

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class HölderTableFunction(AlgebraicFunction):
    """Hölder Table two-dimensional test function.

    A multimodal function with four identical global minima.

    The function is defined as:

    .. math::

        f(x, y) = -|\\sin(\\omega x) \\cos(\\omega y)
        \\exp(|1 - \\frac{\\sqrt{x^2 + y^2}}{\\pi}|)|

    where :math:`\\omega = 1` by default.

    The four global minima are at :math:`f(\\pm 8.05502, \\pm 9.66459) = -19.2085`.

    Parameters
    ----------
    A : float, default=10
        Amplitude parameter.
    angle : float, default=1
        Angular frequency parameter.
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
    >>> from surfaces.test_functions import HölderTableFunction
    >>> func = HölderTableFunction()
    >>> result = func({"x0": 8.05502, "x1": 9.66459})
    """

    name = "Hölder Table Function"
    _name_ = "hölder_table_function"
    __name__ = "HölderTableFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = -19.2085
    x_global = (
        (8.05502, 9.66459),
        (8.05502, -9.66459),
        (-8.05502, 9.66459),
        (-8.05502, -9.66459),
    )

    default_bounds = (-10.0, 10.0)
    n_dim = 2

    latex_formula = r"f(x, y) = -\left|\sin(x)\cos(y)\exp\left(\left|1 - \frac{\sqrt{x^2 + y^2}}{\pi}\right|\right)\right|"
    pgfmath_formula = "-abs(sin(deg(#1))*cos(deg(#2))*exp(abs(1 - sqrt(#1^2 + #2^2)/pi)))"

    # Function sheet attributes
    tagline = (
        "Four deep wells at the corners with a table-like plateau in between. "
        "Tests exploration of disconnected optimal regions."
    )
    display_bounds = (-10.0, 10.0)
    reference = None
    reference_url = "https://www.sfu.ca/~ssurjano/holder.html"

    def __init__(
        self,
        A=10,
        angle=1,
        objective="minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
    ):
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)
        self.n_dim = 2

        self.A = A
        self.angle = angle

    def _create_objective_function(self) -> None:
        def hölder_table_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            loss1 = math.sin(self.angle * x) * math.cos(self.angle * y)
            loss2 = math.exp(abs(1 - (math.sqrt(x**2 + y**2) / math.pi)))

            return -abs(loss1 * loss2)

        self.pure_objective_function = hölder_table_function

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

        loss1 = xp.sin(self.angle * x) * xp.cos(self.angle * y)
        loss2 = xp.exp(xp.abs(1 - xp.sqrt(x**2 + y**2) / math.pi))

        return -xp.abs(loss1 * loss2)

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
