# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class StyblinskiTangFunction(AlgebraicFunction):
    """Styblinski-Tang N-dimensional test function.

    A polynomial function with multiple local minima.

    The function is defined as:

    .. math::

        f(\\vec{x}) = \\frac{1}{2} \\sum_{i=1}^{n} (x_i^4 - 16x_i^2 + 5x_i)

    The global minimum is approximately
    :math:`f(-2.903534, ..., -2.903534) \\approx -39.16617n`.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    default_bounds : tuple
        Default parameter bounds (-5.0, 5.0).

    Examples
    --------
    >>> from surfaces.test_functions import StyblinskiTangFunction
    >>> func = StyblinskiTangFunction(n_dim=2)
    >>> result = func({"x0": -2.903534, "x1": -2.903534})
    """

    name = "Styblinski Tang Function"
    _name_ = "styblinski_tang_function"
    __name__ = "StyblinskiTangFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": True,
        "scalable": True,
    }

    default_bounds = (-5.0, 5.0)

    latex_formula = r"f(\vec{x}) = \frac{1}{2} \sum_{i=1}^{n} \left(x_i^4 - 16x_i^2 + 5x_i\right)"
    pgfmath_formula = "0.5*(#1^4 - 16*#1^2 + 5*#1 + #2^4 - 16*#2^2 + 5*#2)"  # 2D specialization

    # Function sheet attributes
    tagline = (
        "A separable polynomial with two local minima per dimension. "
        "The global minimum lies in an asymmetric valley."
    )
    display_bounds = (-5.0, 5.0)
    display_projection = {"fixed_value": -2.903534}  # Fix all dims except x0, x1 to optimum
    reference = "Styblinski & Tang (1990)"
    reference_url = "https://www.sfu.ca/~ssurjano/stybtang.html"

    def __init__(
        self,
        n_dim: int,
        objective: str = "minimize",
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        noise: Optional["BaseNoise"] = None,
    ) -> None:
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)
        self.n_dim = n_dim
        self.x_global = np.full(n_dim, -2.903534)
        self.f_global = -39.16617 * n_dim

    def _create_objective_function(self) -> None:
        def styblinski_tang_function(params: Dict[str, Any]) -> float:
            loss = 0.0
            for dim in range(self.n_dim):
                dim_str = "x" + str(dim)
                x = params[dim_str]

                loss += x**4 - 16 * x**2 + 5 * x

            return loss / 2

        self.pure_objective_function = styblinski_tang_function

    def _search_space(
        self,
        min: float = -5,
        max: float = 5,
        size: int = 10000,
        value_types: str = "array",
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(min, max, size=size, value_types=value_types)
