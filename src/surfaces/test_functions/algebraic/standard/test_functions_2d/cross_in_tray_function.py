# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class CrossInTrayFunction(AlgebraicFunction):
    """Cross-in-Tray two-dimensional test function.

    A multimodal function with four identical global minima arranged
    symmetrically around the origin.

    The function is defined as:

    .. math::

        f(x, y) = A \\left[|\\sin(\\omega x) \\sin(\\omega y)
        \\exp(|B - \\frac{\\sqrt{x^2+y^2}}{\\pi}|)| + 1\\right]^{0.1}

    where :math:`A = -0.0001`, :math:`B = 100`, and :math:`\\omega = 1` by default.

    The global minima are at :math:`f(\\pm 1.34941, \\pm 1.34941) = -2.06261`.

    Parameters
    ----------
    A : float, default=-0.0001
        Amplitude scaling parameter.
    B : float, default=100
        Exponential base parameter.
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
    >>> from surfaces.test_functions import CrossInTrayFunction
    >>> func = CrossInTrayFunction()
    >>> result = func({"x0": 1.34941, "x1": 1.34941})
    """

    name = "Cross In Tray Function"
    _name_ = "cross_in_tray_function"
    __name__ = "CrossInTrayFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = -2.06261
    x_global = (
        (1.34941, 1.34941),
        (1.34941, -1.34941),
        (-1.34941, 1.34941),
        (-1.34941, -1.34941),
    )

    default_bounds = (-10.0, 10.0)
    n_dim = 2

    latex_formula = r"f(x, y) = -0.0001\left[\left|\sin(x)\sin(y)\exp\left(\left|100 - \frac{\sqrt{x^2+y^2}}{\pi}\right|\right)\right| + 1\right]^{0.1}"
    pgfmath_formula = (
        "-0.0001*(abs(sin(deg(#1))*sin(deg(#2))*exp(abs(100 - sqrt(#1^2 + #2^2)/pi))) + 1)^0.1"
    )

    # Function sheet attributes
    tagline = (
        "Four identical minima at symmetric positions. "
        "A cross-shaped pattern of valleys meets at the origin."
    )
    display_bounds = (-10.0, 10.0)
    reference = None
    reference_url = "https://www.sfu.ca/~ssurjano/crossit.html"

    def __init__(
        self,
        A: float = -0.0001,
        B: float = 100,
        angle: float = 1,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.angle = angle

    def _create_objective_function(self) -> None:
        def cross_in_tray_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            term1 = math.sin(self.angle * x) * math.sin(self.angle * y)
            term2 = math.exp(abs(self.B - (math.sqrt(x**2 + y**2) / math.pi)))

            return self.A * (abs(term1 * term2) + 1) ** 0.1

        self.pure_objective_function = cross_in_tray_function

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

        term1 = xp.sin(self.angle * x) * xp.sin(self.angle * y)
        term2 = xp.exp(xp.abs(self.B - xp.sqrt(x**2 + y**2) / math.pi))

        return self.A * (xp.abs(term1 * term2) + 1) ** 0.1

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
