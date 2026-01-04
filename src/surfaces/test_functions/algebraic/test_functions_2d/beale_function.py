# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class BealeFunction(AlgebraicFunction):
    """Beale two-dimensional test function.

    A multimodal function with sharp peaks at the corners of the input domain.
    It is commonly used for testing optimization algorithms.

    The function is defined as:

    .. math::

        f(x, y) = (A - x + xy)^2 + (B - x + xy^2)^2 + (C - x + xy^3)^2

    where :math:`A = 1.5`, :math:`B = 2.25`, and :math:`C = 2.625` by default.

    The global minimum is :math:`f(3, 0.5) = 0`.

    Parameters
    ----------
    A : float, default=1.5
        First coefficient.
    B : float, default=2.25
        Second coefficient.
    C : float, default=2.625
        Third coefficient.
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    n_dim : int
        Number of dimensions (always 2).
    default_bounds : tuple
        Default parameter bounds (-4.5, 4.5).

    References
    ----------
    .. [1] Global Optimization Test Problems. Retrieved June 2013, from
       http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO

    Examples
    --------
    >>> from surfaces.test_functions import BealeFunction
    >>> func = BealeFunction()
    >>> result = func({"x0": 3.0, "x1": 0.5})
    >>> abs(result) < 1e-10
    True
    """

    name = "Beale Function"
    _name_ = "beale_function"
    __name__ = "BealeFunction"

    _spec = {
        "convex": False,
        "unimodal": True,
        "separable": False,
        "scalable": False,
    }

    f_global = 0.0
    x_global = np.array([3.0, 0.5])

    default_bounds = (-4.5, 4.5)
    n_dim = 2

    latex_formula = r"f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2"
    pgfmath_formula = "(1.5 - #1 + #1*#2)^2 + (2.25 - #1 + #1*#2^2)^2 + (2.625 - #1 + #1*#2^3)^2"

    # Function sheet attributes
    tagline = (
        "Sharp peaks at domain corners with a flat valley leading to the minimum. "
        "Tests navigation through curved valleys."
    )
    display_bounds = (-4.5, 4.5)
    reference = "Beale (1958)"
    reference_url = "https://www.sfu.ca/~ssurjano/beale.html"

    def __init__(
        self,
        A: float = 1.5,
        B: float = 2.25,
        C: float = 2.625,
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
        self.C = C

    def _create_objective_function(self) -> None:
        def beale_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            loss1 = (self.A - x + x * y) ** 2
            loss2 = (self.B - x + x * y**2) ** 2
            loss3 = (self.C - x + x * y**3) ** 2

            return loss1 + loss2 + loss3

        self.pure_objective_function = beale_function

    def _search_space(
        self,
        min: float = -4.5,
        max: float = 4.5,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
