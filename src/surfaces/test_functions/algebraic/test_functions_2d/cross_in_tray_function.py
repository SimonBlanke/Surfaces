# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


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
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

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
    x_global = np.array(
        [
            [1.34941, 1.34941],
            [1.34941, -1.34941],
            [-1.34941, 1.34941],
            [-1.34941, -1.34941],
        ]
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
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        noise: Optional["BaseNoise"] = None,
    ) -> None:
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.angle = angle

    def _create_objective_function(self) -> None:
        def cross_in_tray_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            loss1 = np.sin(self.angle * x) * np.sin(self.angle * y)
            loss2 = np.exp(abs(self.B - (np.sqrt(x**2 + y**2) / np.pi)))

            return self.A * (np.abs(loss1 * loss2) + 1) ** 0.1

        self.pure_objective_function = cross_in_tray_function

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
