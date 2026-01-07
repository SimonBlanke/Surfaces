# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


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
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

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
    x_global = (0.0, 0.0)

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
        angle: float = 2 * math.pi,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

        self.n_dim = 2

        self.A = A
        self.angle = angle

    def _create_objective_function(self) -> None:
        def ackley_function(params: Dict[str, Any]) -> float:
            x = params["x0"]
            y = params["x1"]

            term1 = -self.A * math.exp(-0.2 * math.sqrt(0.5 * (x * x + y * y)))
            term2 = -math.exp(0.5 * (math.cos(self.angle * x) + math.cos(self.angle * y)))

            return term1 + term2 + math.e + self.A

        self.pure_objective_function = ackley_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation.

        Parameters
        ----------
        X : ArrayLike
            Array of shape (n_points, 2).

        Returns
        -------
        ArrayLike
            Array of shape (n_points,).
        """
        xp = get_array_namespace(X)

        x = X[:, 0]
        y = X[:, 1]

        # f(x,y) = -A*exp(-0.2*sqrt(0.5*(x^2+y^2)))
        #          - exp(0.5*(cos(angle*x) + cos(angle*y))) + e + A
        term1 = -self.A * xp.exp(-0.2 * xp.sqrt(0.5 * (x**2 + y**2)))
        term2 = -xp.exp(0.5 * (xp.cos(self.angle * x) + xp.cos(self.angle * y)))

        return term1 + term2 + math.e + self.A

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
