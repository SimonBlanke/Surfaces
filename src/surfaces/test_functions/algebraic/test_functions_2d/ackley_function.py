# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class AckleyFunction(AlgebraicFunction):
    """Ackley two-dimensional test function.

    A non-convex function used as a performance test problem for optimization
    algorithms. It has a nearly flat outer region with a large hole at the
    center, making it challenging for optimization methods.

    The function is defined as:

    .. math::

        f(x, y) = -A \\exp\\left[-0.2\\sqrt{0.5(x^2+y^2)}\\right]
        - \\exp\\left[0.5(\\cos \\omega x + \\cos \\omega y)\\right] + e + A

    where :math:`A = 20` and :math:`\\omega = 2\\pi` by default.

    The global minimum is :math:`f(0, 0) = 0`.

    Parameters
    ----------
    A : float, default=20
        Amplitude parameter.
    angle : float, default=2*pi
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
        Default parameter bounds (-5.0, 5.0).

    References
    ----------
    .. [1] Ackley, D. H. (1987). "A connectionist machine for genetic
       hillclimbing". Kluwer Academic Publishers, Boston MA.

    Examples
    --------
    >>> from surfaces.test_functions import AckleyFunction
    >>> func = AckleyFunction()
    >>> result = func({"x0": 0.0, "x1": 0.0})
    >>> abs(result) < 1e-10
    True
    """

    name = "Ackley Function"
    _name_ = "ackley_function"
    __name__ = "AckleyFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = 0.0
    x_global = np.array([0.0, 0.0])

    default_bounds = (-5.0, 5.0)
    n_dim = 2

    latex_formula = r"f(x, y) = -20\exp\left[-0.2\sqrt{0.5(x^2+y^2)}\right] - \exp\left[0.5(\cos 2\pi x + \cos 2\pi y)\right] + e + 20"
    pgfmath_formula = "-20*exp(-0.2*sqrt(0.5*(#1^2 + #2^2))) - exp(0.5*(cos(deg(2*pi*#1)) + cos(deg(2*pi*#2)))) + exp(1) + 20"

    # Function sheet attributes
    tagline = (
        "A nearly flat outer region surrounds a steep funnel at the center. "
        "Tests an optimizer's ability to escape deceptive plateaus."
    )
    display_bounds = (-5.0, 5.0)
    reference = "Ackley (1987)"
    reference_url = "https://www.sfu.ca/~ssurjano/ackley.html"

    def __init__(
        self,
        A: float = 20,
        angle: float = 2 * np.pi,
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
        self.angle = angle

    def _create_objective_function(self) -> None:
        def ackley_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            loss1 = -self.A * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
            loss2 = -np.exp(0.5 * (np.cos(self.angle * x) + np.cos(self.angle * y)))
            loss3 = np.exp(1)
            loss4 = self.A

            return loss1 + loss2 + loss3 + loss4

        self.pure_objective_function = ackley_function

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
