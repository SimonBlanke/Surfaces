# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class EggholderFunction(AlgebraicFunction):
    """Eggholder two-dimensional test function.

    A difficult optimization test problem due to its many local minima.

    The function is defined as:

    .. math::

        f(x, y) = -(y + 47) \\sin\\sqrt{|\\frac{x}{2} + (y + 47)|}
        - x \\sin\\sqrt{|x - (y + 47)|}

    The global minimum is :math:`f(512, 404.2319) = -959.6407`.

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
        Default parameter bounds (-1000.0, 1000.0).

    Examples
    --------
    >>> from surfaces.test_functions import EggholderFunction
    >>> func = EggholderFunction()
    >>> result = func({"x0": 512.0, "x1": 404.2319})
    """

    name = "Eggholder Function"
    _name_ = "eggholder_function"
    __name__ = "EggholderFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = -959.6407
    x_global = np.array([512.0, 404.2319])

    default_bounds = (-1000.0, 1000.0)
    n_dim = 2

    latex_formula = r"f(x, y) = -(y + 47)\sin\sqrt{\left|\frac{x}{2} + (y + 47)\right|} - x\sin\sqrt{|x - (y + 47)|}"
    pgfmath_formula = (
        "-(#2 + 47)*sin(deg(sqrt(abs(#1/2 + (#2 + 47))))) - #1*sin(deg(sqrt(abs(#1 - (#2 + 47)))))"
    )

    # Function sheet attributes
    tagline = (
        "A rugged landscape with many deep valleys and sharp ridges. "
        "One of the most challenging benchmark functions."
    )
    display_bounds = (-512.0, 512.0)
    reference = None
    reference_url = "https://www.sfu.ca/~ssurjano/egg.html"

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
        def eggholder_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(
                np.sqrt(np.abs(x - (y + 47)))
            )

        self.pure_objective_function = eggholder_function

    def _search_space(
        self,
        min: float = -1000,
        max: float = 1000,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
