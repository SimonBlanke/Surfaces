# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class RosenbrockFunction(AlgebraicFunction):
    """Rosenbrock N-dimensional test function.

    Also known as the "banana function" due to the shape of its contour
    lines. It is a classic optimization test problem with a narrow,
    curved valley leading to the global minimum.

    The function is defined as:

    .. math::

        f(\\vec{x}) = \\sum_{i=1}^{n-1} [B(x_{i+1} - x_i^2)^2 + (A - x_i)^2]

    where :math:`A = 1` and :math:`B = 100` by default.

    The global minimum is :math:`f(\\vec{1}) = 0`.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    A : float, default=1
        First coefficient.
    B : float, default=100
        Second coefficient controlling the steepness of the valley.
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

    References
    ----------
    .. [1] Rosenbrock, H.H. (1960). "An automatic method for finding the
       greatest or least value of a function". The Computer Journal.
       3 (3): 175-184.

    Examples
    --------
    >>> from surfaces.test_functions import RosenbrockFunction
    >>> func = RosenbrockFunction(n_dim=2)
    >>> result = func({"x0": 1.0, "x1": 1.0})
    >>> abs(result) < 1e-10
    True
    """

    name = "Rosenbrock Function"
    _name_ = "rosenbrock_function"
    __name__ = "RosenbrockFunction"

    _spec = {
        "convex": False,
        "unimodal": True,
        "separable": False,
        "scalable": True,
    }

    f_global = 0.0

    default_bounds = (-5.0, 5.0)

    latex_formula = (
        r"f(\vec{x}) = \sum_{i=1}^{n-1} \left[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\right]"
    )
    pgfmath_formula = "100*(#2 - #1^2)^2 + (1 - #1)^2"  # 2D specialization

    # Function sheet attributes
    tagline = (
        "The classic banana-shaped valley. A narrow curved ridge leads to the minimum, "
        "making convergence notoriously slow."
    )
    display_bounds = (-2.0, 2.0)
    display_projection = {"fixed_value": 1.0}  # Fix all dims except x0, x1 to optimum
    reference = "Rosenbrock (1960)"
    reference_url = "https://doi.org/10.1093/comjnl/3.3.175"

    def __init__(
        self,
        n_dim: int,
        A: float = 1,
        B: float = 100,
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

        self.A = A
        self.B = B
        self.x_global = np.ones(n_dim)

    def _create_objective_function(self) -> None:
        def rosenbrock_function(params: Dict[str, Any]) -> float:
            loss = 0.0
            for dim in range(self.n_dim - 1):
                dim_str = "x" + str(dim)
                dim_str_1 = "x" + str(dim + 1)

                x = params[dim_str]
                y = params[dim_str_1]

                loss += (self.A - x) ** 2 + self.B * (y - x**2) ** 2
            return loss

        self.pure_objective_function = rosenbrock_function

    def _search_space(
        self,
        min: float = -5,
        max: float = 5,
        size: int = 10000,
        value_types: str = "array",
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(min, max, size=size, value_types=value_types)
