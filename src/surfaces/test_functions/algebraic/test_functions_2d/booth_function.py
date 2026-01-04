# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class BoothFunction(AlgebraicFunction):
    """Booth two-dimensional test function.

    A simple polynomial optimization test function.

    The function is defined as:

    .. math::

        f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2

    The global minimum is :math:`f(1, 3) = 0`.

    Parameters
    ----------
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

    References
    ----------
    .. [1] Global Optimization Test Problems. Retrieved June 2013, from
       http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO

    Examples
    --------
    >>> from surfaces.test_functions import BoothFunction
    >>> func = BoothFunction()
    >>> result = func({"x0": 1.0, "x1": 3.0})
    """

    name = "Booth Function"
    _name_ = "booth_function"
    __name__ = "BoothFunction"

    _spec = {
        "convex": False,
        "unimodal": True,
        "separable": False,
        "scalable": False,
    }

    f_global = 0.0
    x_global = np.array([1.0, 3.0])

    default_bounds = (-10.0, 10.0)
    n_dim = 2

    latex_formula = r"f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2"
    pgfmath_formula = "(#1 + 2*#2 - 7)^2 + (2*#1 + #2 - 5)^2"

    # Function sheet attributes
    tagline = (
        "A simple polynomial with a single minimum in a curved valley. "
        "Often used as a baseline test for optimization algorithms."
    )
    display_bounds = (-10.0, 10.0)
    reference = None
    reference_url = "https://www.sfu.ca/~ssurjano/booth.html"

    def __init__(
        self,
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

    def _create_objective_function(self) -> None:
        def booth_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            loss1 = (x + 2 * y - 7) ** 2
            loss2 = (2 * x + y - 5) ** 2

            return loss1 + loss2

        self.pure_objective_function = booth_function

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
