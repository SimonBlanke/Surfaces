# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class RastriginFunction(AlgebraicFunction):
    """Rastrigin N-dimensional test function.

    A highly multimodal function with many local minima arranged in a
    regular lattice pattern. It is commonly used to test the ability
    of optimization algorithms to escape local optima.

    The function is defined as:

    .. math::

        f(\\vec{x}) = An + \\sum_{i=1}^{n} [x_i^2 - A\\cos(\\omega x_i)]

    where :math:`A = 10` and :math:`\\omega = 2\\pi` by default.

    The global minimum is :math:`f(\\vec{0}) = 0`.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    A : float, default=10
        Amplitude of the cosine modulation.
    angle : float, default=2*pi
        Angular frequency parameter.
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
    >>> from surfaces.test_functions import RastriginFunction
    >>> func = RastriginFunction(n_dim=2)
    >>> result = func({"x0": 0.0, "x1": 0.0})
    >>> abs(result) < 1e-10
    True
    """

    name = "Rastrigin Function"
    _name_ = "rastrigin_function"
    __name__ = "RastriginFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": True,
        "scalable": True,
    }

    f_global = 0.0

    default_bounds = (-5.0, 5.0)

    latex_formula = r"f(\vec{x}) = 10n + \sum_{i=1}^{n} \left[x_i^2 - 10\cos(2\pi x_i)\right]"
    pgfmath_formula = (
        "20 + #1^2 - 10*cos(deg(2*pi*#1)) + #2^2 - 10*cos(deg(2*pi*#2))"  # 2D specialization
    )

    # Function sheet attributes
    tagline = (
        "A highly multimodal landscape with regularly spaced local minima. "
        "The cosine modulation creates a challenging grid of traps."
    )
    display_bounds = (-5.12, 5.12)
    display_projection = {"fixed_value": 0.0}  # Fix all dims except x0, x1 to 0
    reference = "Rastrigin (1974)"
    reference_url = "https://www.sfu.ca/~ssurjano/rastr.html"

    def __init__(
        self,
        n_dim: int,
        A: float = 10,
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

        self.n_dim = n_dim
        self.A = A
        self.angle = angle
        self.x_global = np.zeros(n_dim)

    def _create_objective_function(self) -> None:
        def rastrigin_function(params: Dict[str, Any]) -> float:
            loss = 0.0
            for dim in range(self.n_dim):
                dim_str = "x" + str(dim)
                x = params[dim_str]

                loss += x * x - self.A * np.cos(self.angle * x)

            return self.A * self.n_dim + loss

        self.pure_objective_function = rastrigin_function

    def _search_space(
        self,
        min: float = -5,
        max: float = 5,
        size: int = 10000,
        value_types: str = "array",
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(min, max, size=size, value_types=value_types)
