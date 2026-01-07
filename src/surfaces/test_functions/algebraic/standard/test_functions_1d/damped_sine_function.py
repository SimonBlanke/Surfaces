# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class DampedSineFunction(AlgebraicFunction):
    """Damped Sine one-dimensional test function.

    A one-dimensional test function combining sine and linear terms
    with Gaussian damping. The exponential decay creates a localized
    oscillatory region near the origin.

    The function is defined as:

    .. math::

        f(x) = -(x + \\sin(x)) e^{-x^2}

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
        Default parameter bounds (-10, 10).

    References
    ----------
    .. [1] AMPGO (Adaptive Memory Programming for Global Optimization)
       benchmark suite, Problem06.
       http://infinity77.net/global_optimization/test_functions_1d.html

    .. [2] Gavana, A. (2013). "Global Optimization Benchmarks and AMPGO".

    Examples
    --------
    >>> from surfaces.test_functions import DampedSineFunction
    >>> func = DampedSineFunction()
    >>> func({"x0": 0.6796})  # Near global minimum
    -0.824...
    >>> search_space = func.search_space
    >>> len(search_space)
    1
    """

    name = "Damped Sine Function"
    _name_ = "damped_sine_function"
    __name__ = "DampedSineFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": True,
        "scalable": False,
    }

    f_global = -0.8242393984760573
    x_global = (0.6795787635255166,)

    default_bounds = (-10.0, 10.0)
    n_dim = 1

    latex_formula = r"f(x) = -(x + \sin(x)) e^{-x^2}"
    pgfmath_formula = "-(#1 + sin(deg(#1))) * exp(-#1^2)"

    # Function sheet attributes
    tagline = (
        "Oscillatory behavior with Gaussian damping. The exponential decay "
        "localizes activity near the origin while outer regions flatten."
    )
    display_bounds = (-3.0, 3.0)
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
        def damped_sine_function(params: Dict[str, Any]) -> float:
            x = params["x0"]

            return -(x + math.sin(x)) * math.exp(-(x**2))

        self.pure_objective_function = damped_sine_function

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
        return -(x + xp.sin(x)) * xp.exp(-(x**2))

    def _search_space(
        self,
        min: float = -10.0,
        max: float = 10.0,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
