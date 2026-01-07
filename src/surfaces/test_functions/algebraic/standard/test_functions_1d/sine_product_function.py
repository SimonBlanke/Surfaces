# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class SineProductFunction(AlgebraicFunction):
    """Sine Product one-dimensional test function.

    A one-dimensional test function that multiplies a linear term
    with a sine function. The amplitude of oscillations grows with x,
    creating increasingly deep valleys at larger values.

    The function is defined as:

    .. math::

        f(x) = -x \\sin(x)

    Parameters
    ----------
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Attributes
    ----------
    n_dim : int
        Number of dimensions (always 1).
    default_bounds : tuple
        Default parameter bounds (0, 10).

    References
    ----------
    .. [1] AMPGO (Adaptive Memory Programming for Global Optimization)
       benchmark suite, Problem10.
       http://infinity77.net/global_optimization/test_functions_1d.html

    .. [2] Gavana, A. (2013). "Global Optimization Benchmarks and AMPGO".

    Examples
    --------
    >>> from surfaces.test_functions import SineProductFunction
    >>> func = SineProductFunction()
    >>> func({"x0": 7.9787})  # Near global minimum
    -7.916...
    >>> search_space = func.search_space
    >>> len(search_space)
    1
    """

    name = "Sine Product Function"
    _name_ = "sine_product_function"
    __name__ = "SineProductFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": True,
        "scalable": False,
    }

    f_global = -7.916727371587256
    x_global = (7.9786653537049483,)

    default_bounds = (0.0, 10.0)
    n_dim = 1

    latex_formula = r"f(x) = -x \sin(x)"
    pgfmath_formula = "-#1 * sin(deg(#1))"

    # Function sheet attributes
    tagline = (
        "Growing oscillations with increasing amplitude. The deepest valleys "
        "appear at larger x values where the linear term dominates."
    )
    display_bounds = (0.0, 10.0)
    reference = "Gavana (2013)"
    reference_url = "http://infinity77.net/global_optimization/test_functions_1d.html"

    def __init__(
        self,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)
        self.n_dim = 1

    def _create_objective_function(self) -> None:
        def sine_product_function(params: Dict[str, Any]) -> float:
            x = params["x0"]

            return -x * math.sin(x)

        self.pure_objective_function = sine_product_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation.

        Parameters
        ----------
        X : ArrayLike
            Array of shape (n_points, 1).

        Returns
        -------
        ArrayLike
            Array of shape (n_points,).
        """
        xp = get_array_namespace(X)

        x = X[:, 0]
        return -x * xp.sin(x)

    def _search_space(
        self,
        min: float = 0.0,
        max: float = 10.0,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
