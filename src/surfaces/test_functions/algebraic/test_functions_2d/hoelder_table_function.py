# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


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
    x_global = np.array(
        [
            [8.05502, 9.66459],
            [8.05502, -9.66459],
            [-8.05502, 9.66459],
            [-8.05502, -9.66459],
        ]
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
        sleep=0,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
        noise=None,
    ):
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)
        self.n_dim = 2

        self.A = A
        self.angle = angle

    def _create_objective_function(self) -> None:
        def hölder_table_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            loss1 = np.sin(self.angle * x) * np.cos(self.angle * y)
            loss2 = np.exp(abs(1 - (np.sqrt(x**2 + y**2) / np.pi)))

            return -np.abs(loss1 * loss2)

        self.pure_objective_function = hölder_table_function

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
