# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class ForresterFunction(AlgebraicFunction):
    """Forrester one-dimensional test function.

    A one-dimensional test function commonly used for testing surrogate
    modeling and multi-fidelity optimization methods. The function is
    multimodal with one global minimum, one local minimum, and a
    zero-gradient inflection point.

    The function is defined as:

    .. math::

        f(x) = (6x - 2)^2 \\sin(12x - 4)

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
        Default parameter bounds (0, 1).

    References
    ----------
    .. [1] Forrester, A., Sobester, A., & Keane, A. (2008). "Engineering
       design via surrogate modelling: a practical guide". Wiley.

    .. [2] Virtual Library of Simulation Experiments (VLSE):
       https://www.sfu.ca/~ssurjano/forretal08.html

    Examples
    --------
    >>> from surfaces.test_functions import ForresterFunction
    >>> func = ForresterFunction()
    >>> func({"x0": 0.757249})  # Near global minimum
    -6.020740...
    >>> search_space = func.search_space
    >>> len(search_space)
    1
    """

    name = "Forrester Function"
    _name_ = "forrester_function"
    __name__ = "ForresterFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": True,
        "scalable": False,
    }

    f_global = -6.020740055766075
    x_global = (0.7572487144081974,)

    default_bounds = (0.0, 1.0)
    n_dim = 1

    latex_formula = r"f(x) = (6x - 2)^2 \sin(12x - 4)"
    pgfmath_formula = "(6*#1 - 2)^2 * sin(deg(12*#1 - 4))"

    # Function sheet attributes
    tagline = (
        "A classic surrogate modeling benchmark. One global minimum, "
        "one local minimum, and a zero-gradient inflection point within a compact domain."
    )
    display_bounds = (0.0, 1.0)
    reference = "Forrester et al. (2008)"
    reference_url = "https://www.sfu.ca/~ssurjano/forretal08.html"

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
        def forrester_function(params: Dict[str, Any]) -> float:
            x = params["x0"]

            return ((6 * x - 2) ** 2) * math.sin(12 * x - 4)

        self.pure_objective_function = forrester_function

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
        return ((6 * x - 2) ** 2) * xp.sin(12 * x - 4)

    def _search_space(
        self,
        min: float = 0.0,
        max: float = 1.0,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
