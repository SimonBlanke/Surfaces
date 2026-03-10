from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class ColvilleFunction(AlgebraicFunction):
    """Colville 4-dimensional test function.

    A multimodal, non-convex, continuous function fixed at 4 dimensions.

    The function is defined as:

    .. math::

        f(\\vec{x}) = 100(x_1^2 - x_2)^2 + (x_1 - 1)^2
        + (x_3 - 1)^2 + 90(x_3^2 - x_4)^2
        + 10.1((x_2 - 1)^2 + (x_4 - 1)^2)
        + 19.8(x_2 - 1)(x_4 - 1)

    The global minimum is :math:`f(1, 1, 1, 1) = 0`.

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.
    memory : bool, default=False
        Whether to store function evaluations in memory.
    collect_data : bool, default=True
        Whether to collect data for visualization/history.
    callbacks : list of callable, optional
        Functions to call after evaluation.
    catch_errors : dict, optional
        Dictionary mapping error types to return values.

    Attributes
    ----------
    n_dim : int
        Number of dimensions (fixed at 4).

    Examples
    --------
    >>> from surfaces.test_functions import ColvilleFunction
    >>> func = ColvilleFunction()
    >>> result = func({"x0": 1.0, "x1": 1.0, "x2": 1.0, "x3": 1.0})
    >>> abs(float(result)) < 1e-10
    True
    """

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
        "default_bounds": (-10.0, 10.0),
    }

    f_global = 0.0

    latex_formula = (
        r"f(\vec{x}) = 100(x_1^2 - x_2)^2 + (x_1 - 1)^2 + (x_3 - 1)^2 "
        r"+ 90(x_3^2 - x_4)^2 + 10.1((x_2 - 1)^2 + (x_4 - 1)^2) "
        r"+ 19.8(x_2 - 1)(x_4 - 1)"
    )
    pgfmath_formula = None

    tagline = "A 4-dimensional polynomial function with a narrow curved valley."
    display_bounds = (-10.0, 10.0)

    display_projection = {"fixed_values": {"x2": 1.0, "x3": 1.0}}

    reference = "Colville, A. R. (1968). A comparative study on nonlinear programming codes."
    reference_url = "https://www.sfu.ca/~ssurjano/colville.html"

    def __init__(
        self,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        self.n_dim = 4
        self.x_global = (1.0, 1.0, 1.0, 1.0)

        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

    def _objective(self, params: Dict[str, Any]) -> float:
        x0 = params["x0"]
        x1 = params["x1"]
        x2 = params["x2"]
        x3 = params["x3"]

        result = (
            100.0 * (x0**2 - x1) ** 2
            + (x0 - 1.0) ** 2
            + (x2 - 1.0) ** 2
            + 90.0 * (x2**2 - x3) ** 2
            + 10.1 * ((x1 - 1.0) ** 2 + (x3 - 1.0) ** 2)
            + 19.8 * (x1 - 1.0) * (x3 - 1.0)
        )

        return float(result)

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation for Colville Function."""
        xp = get_array_namespace(X)

        x0 = X[:, 0]
        x1 = X[:, 1]
        x2 = X[:, 2]
        x3 = X[:, 3]

        result = (
            100.0 * (x0**2 - x1) ** 2
            + (x0 - 1.0) ** 2
            + (x2 - 1.0) ** 2
            + 90.0 * (x2**2 - x3) ** 2
            + 10.1 * ((x1 - 1.0) ** 2 + (x3 - 1.0) ** 2)
            + 19.8 * (x1 - 1.0) * (x3 - 1.0)
        )

        return result

    def _search_space(
        self,
        min: float = -10.0,
        max: float = 10.0,
        size: int = 10000,
        value_types: str = "array",
    ) -> Dict[str, Any]:
        space_1d = super()._create_n_dim_search_space(min, max, size=size, value_types=value_types)[
            "x0"
        ]

        return {f"x{i}": space_1d.copy() for i in range(4)}
