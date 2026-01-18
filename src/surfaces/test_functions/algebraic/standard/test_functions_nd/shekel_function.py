# Author: Zohaib Hassan

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class ShekelFunction(AlgebraicFunction):
    """Shekel 4-dimensional test function.

    A multimodal, non-convex, continuous function. It is defined as the
    sum of m inverse quadratic functions.

    The function is defined as:

    .. math::

        f(\\vec{x}) = - \\sum_{i=1}^{m} \\left( \\sum_{j=1}^{4} (x_j - a_{ij})^2 + c_i \\right)^{-1}

    where :math:`m` is the number of maxima (typically 5, 7, or 10).

    The global minimum is located at :math:`x \\approx (4, 4, 4, 4)` and
    the value depends on :math:`m`.

    Parameters
    ----------
    m : int, default=10
        Number of maxima. Standard values are 5, 7, or 10.
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.
    validate : bool, default=True
        Whether to validate parameters against the search space.

    Attributes
    ----------
    n_dim : int
        Number of dimensions (fixed at 4 for standard Shekel).
    default_bounds : tuple
        Default parameter bounds (0.0, 10.0).

    Examples
    --------
    >>> from surfaces.test_functions import ShekelFunction
    >>> func = ShekelFunction(m=10)
    >>> result = func({"x0": 4.0, "x1": 4.0, "x2": 4.0, "x3": 4.0})
    >>> float(result) < -10.0
    True
    >>> len(func.search_space)
    4
    """

    name = "Shekel Function"
    _name_ = "shekel_function"
    __name__ = "ShekelFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = -10.536
    default_bounds = (0.0, 10.0)

    latex_formula = r"f(\vec{x}) = - \sum_{i=1}^{m} \left( \sum_{j=1}^{4} (x_j - a_{ij})^2 + c_i \right)^{-1}"
    pgfmath_formula = None

    # Function sheet attributes
    tagline = (
        "A multimodal function with m sharp peaks. Often called 'Foxholes', "
        "it tests an optimizer's ability to find a global minimum among many locals."
    )
    display_bounds = (0.0, 10.0)
    display_projection = {"fixed_values": {"x2": 4.0, "x3": 4.0}}
    reference = "Shekel, J. (1971). Test function for multivariate search problems."
    reference_url = "https://www.sfu.ca/~ssurjano/shekel.html"

    def __init__(
            self,
            m: int = 10,
            objective: str = "minimize",
            modifiers: Optional[List[BaseModifier]] = None,
            memory: bool = False,
            collect_data: bool = True,
            callbacks: Optional[Union[Callable, List[Callable]]] = None,
            catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)
        self.n_dim = 4
        self.m = m

        self.A = np.array([
            [4.0, 4.0, 4.0, 4.0],
            [1.0, 1.0, 1.0, 1.0],
            [8.0, 8.0, 8.0, 8.0],
            [6.0, 6.0, 6.0, 6.0],
            [3.0, 7.0, 3.0, 7.0],
            [2.0, 9.0, 2.0, 9.0],
            [5.0, 5.0, 3.0, 3.0],
            [8.0, 1.0, 8.0, 1.0],
            [6.0, 2.0, 6.0, 2.0],
            [7.0, 3.6, 7.0, 3.6],
        ])

        self.c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

        if m < 10:
            self.A = self.A[:m]
            self.c = self.c[:m]

        self.x_global = (4.0, 4.0, 4.0, 4.0)

    def _create_objective_function(self) -> None:
        def shekel_function(params: Dict[str, Any]) -> float:
            x_input = np.array([params[f"x{i}"] for i in range(self.n_dim)])

            result = 0.0
            for i in range(self.m):
                # (x - a_i)^T (x - a_i)
                diff = x_input - self.A[i]
                sq_sum = np.dot(diff, diff)

                result -= 1.0 / (sq_sum + self.c[i])

            return result

        self.pure_objective_function = shekel_function

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation.

        Parameters
        ----------
        X : ArrayLike
            Array of shape (n_points, n_dim).

        Returns
        -------
        ArrayLike
            Array of shape (n_points,).
        """
        xp = get_array_namespace(X)

        A = xp.asarray(self.A)
        c = xp.asarray(self.c)

        n_points = X.shape[0]
        result = xp.zeros(n_points)

        for i in range(self.m):
            diff = X - A[i]

            sq_sum = xp.sum(diff ** 2, axis=1)

            result -= 1.0 / (sq_sum + c[i])

        return result

    def _search_space(
            self,
            min: float = 0.0,
            max: float = 10.0,
            size: int = 10000,
            value_types: str = "array",
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(min, max, size=size, value_types=value_types)