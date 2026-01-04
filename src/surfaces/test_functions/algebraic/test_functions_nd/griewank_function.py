# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class GriewankFunction(AlgebraicFunction):
    """Griewank N-dimensional test function.

    A multimodal function with many regularly distributed local minima.
    The number of local minima increases with dimensionality.

    The function is defined as:

    .. math::

        f(\\vec{x}) = \\sum_{i=1}^{n} \\frac{x_i^2}{4000}
        - \\prod_{i=1}^{n} \\cos\\left(\\frac{x_i}{\\sqrt{i}}\\right) + 1

    The global minimum is :math:`f(\\vec{0}) = 0`.

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
        Default parameter bounds (-100.0, 100.0).

    Examples
    --------
    >>> from surfaces.test_functions import GriewankFunction
    >>> func = GriewankFunction(n_dim=2)
    >>> result = func({"x0": 0.0, "x1": 0.0})
    >>> abs(result) < 1e-10
    True
    """

    name = "Griewank Function"
    _name_ = "griewank_function"
    __name__ = "GriewankFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": True,
    }

    f_global = 0.0

    default_bounds = (-100.0, 100.0)

    latex_formula = r"f(\vec{x}) = \sum_{i=1}^{n} \frac{x_i^2}{4000} - \prod_{i=1}^{n} \cos\left(\frac{x_i}{\sqrt{i}}\right) + 1"
    pgfmath_formula = (
        "#1^2/4000 + #2^2/4000 - cos(deg(#1))*cos(deg(#2/sqrt(2))) + 1"  # 2D specialization
    )

    # Function sheet attributes
    tagline = (
        "A product of cosines superimposed on a parabolic bowl. "
        "Local minima become less pronounced at larger scales."
    )
    display_bounds = (-10.0, 10.0)
    display_projection = {"fixed_value": 0.0}  # Fix all dims except x0, x1 to 0
    reference = "Griewank (1981)"
    reference_url = "https://www.sfu.ca/~ssurjano/griewank.html"

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
        self.x_global = np.zeros(n_dim)

    def _create_objective_function(self) -> None:
        def griewank_function(params: Dict[str, Any]) -> float:
            loss_sum = 0.0
            loss_product = 1.0
            for dim in range(self.n_dim):
                dim_str = "x" + str(dim)
                x = params[dim_str]

                loss_sum += x**2 / 4000
                loss_product *= np.cos(x / np.sqrt(dim + 1))

            return loss_sum - loss_product + 1

        self.pure_objective_function = griewank_function

    def _search_space(
        self,
        min: float = -100,
        max: float = 100,
        size: int = 10000,
        value_types: str = "array",
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(min, max, size=size, value_types=value_types)
