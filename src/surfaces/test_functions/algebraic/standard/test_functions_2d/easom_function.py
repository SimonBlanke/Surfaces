# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Dict, List, Optional

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class EasomFunction(AlgebraicFunction):
    """Easom two-dimensional test function.

    A unimodal function with a very small area relative to the search space
    where the function has a significant gradient. The global minimum has
    a small basin of attraction.

    The function is defined as:

    .. math::

        f(x, y) = A \\cos(\\omega x) \\cos(\\omega y)
        \\exp\\left[-(x - \\pi/B)^2 - (y - \\pi/B)^2\\right]

    where :math:`A = -1`, :math:`B = 1`, and :math:`\\omega = 1` by default.

    The global minimum is :math:`f(\\pi, \\pi) = -1`.

    Parameters
    ----------
    A : float, default=-1
        Amplitude parameter.
    B : float, default=1
        Scaling parameter for the optimum location.
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
    >>> from surfaces.test_functions import EasomFunction
    >>> import numpy as np
    >>> func = EasomFunction()
    >>> result = func({"x0": np.pi, "x1": np.pi})
    >>> abs(result + 1.0) < 1e-10
    True
    """

    name = "Easom Function"
    _name_ = "easom_function"
    __name__ = "EasomFunction"

    _spec = {
        "convex": False,
        "unimodal": True,
        "separable": True,
        "scalable": False,
    }

    f_global = -1.0
    x_global = (math.pi, math.pi)

    default_bounds = (-10.0, 10.0)
    n_dim = 2

    latex_formula = r"f(x, y) = -\cos(x)\cos(y)\exp\left[-(x - \pi)^2 - (y - \pi)^2\right]"
    pgfmath_formula = "-cos(deg(#1))*cos(deg(#2))*exp(-((#1 - pi)^2 + (#2 - pi)^2))"

    # Function sheet attributes
    tagline = (
        "A single sharp peak in a vast flat landscape. "
        "The tiny basin of attraction makes this notoriously difficult."
    )
    display_bounds = (0.0, 6.0)
    reference = "Easom (1990)"
    reference_url = "https://www.sfu.ca/~ssurjano/easom.html"

    def __init__(
        self,
        A=-1,
        B=1,
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
        self.B = B
        self.angle = angle

    def _create_objective_function(self) -> None:
        def easom_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            loss1 = self.A * math.cos(x * self.angle) * math.cos(y * self.angle)
            loss2 = math.exp(-((x - math.pi / self.B) ** 2 + (y - math.pi / self.B) ** 2))

            return loss1 * loss2

        self.pure_objective_function = easom_function

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

        loss1 = self.A * xp.cos(x * self.angle) * xp.cos(y * self.angle)
        loss2 = xp.exp(-((x - math.pi / self.B) ** 2 + (y - math.pi / self.B) ** 2))

        return loss1 * loss2

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
