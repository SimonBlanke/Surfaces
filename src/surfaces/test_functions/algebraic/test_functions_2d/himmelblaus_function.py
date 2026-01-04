# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    pass


class HimmelblausFunction(AlgebraicFunction):
    """Himmelblau's two-dimensional test function.

    A multimodal function with four identical global minima.

    The function is defined as:

    .. math::

        f(x, y) = (x^2 + y + A)^2 + (x + y^2 + B)^2

    where :math:`A = -11` and :math:`B = -7` by default.

    The four global minima are:
        - :math:`f(3.0, 2.0) = 0`
        - :math:`f(-2.805118, 3.131312) = 0`
        - :math:`f(-3.779310, -3.283186) = 0`
        - :math:`f(3.584428, -1.848126) = 0`

    Parameters
    ----------
    A : float, default=-11
        First constant term.
    B : float, default=-7
        Second constant term.
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

    Examples
    --------
    >>> from surfaces.test_functions import HimmelblausFunction
    >>> func = HimmelblausFunction()
    >>> result = func({"x0": 3.0, "x1": 2.0})
    >>> abs(result) < 1e-10
    True
    """

    name = "Himmelblau's Function"
    _name_ = "himmelblaus_function"
    __name__ = "HimmelblausFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = 0.0
    x_global = np.array(
        [
            [3.0, 2.0],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126],
        ]
    )

    default_bounds = (-5.0, 5.0)
    n_dim = 2

    latex_formula = r"f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2"
    pgfmath_formula = "(#1^2 + #2 - 11)^2 + (#1 + #2^2 - 7)^2"

    # Function sheet attributes
    tagline = (
        "Four symmetric valleys leading to equally optimal solutions. "
        "A classic test for multi-solution discovery algorithms."
    )
    display_bounds = (-6.0, 6.0)
    reference = "Himmelblau (1972)"
    reference_url = "https://doi.org/10.1002/aic.690180227"

    def __init__(
        self,
        A=-11,
        B=-7,
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
        self.B = B

    def _create_objective_function(self) -> None:
        def himmelblaus_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            loss1 = (x**2 + y + self.A) ** 2
            loss2 = (x + y**2 + self.B) ** 2

            return loss1 + loss2

        self.pure_objective_function = himmelblaus_function

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
