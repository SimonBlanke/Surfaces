# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from .._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    pass


class SimionescuFunction(AlgebraicFunction):
    """Simionescu two-dimensional constrained test function.

    A function with a bumpy constraint boundary. Points outside the
    constraint region return NaN.

    The function is defined as:

    .. math::

        f(x, y) = Axy

    Subject to:

    .. math::

        x^2 + y^2 \\le [r_T + r_S \\cos(n \\arctan(x/y))]^2

    where :math:`A = 0.1`, :math:`r_T = 1`, :math:`r_S = 0.2`, and :math:`n = 8`
    by default.

    The global minimum is :math:`f(\\pm 0.84852813, \\mp 0.84852813) = -0.072`.

    Parameters
    ----------
    A : float, default=0.1
        Amplitude scaling parameter.
    r_T : float, default=1
        Constraint radius parameter.
    r_S : float, default=0.2
        Constraint wave amplitude.
    n : int, default=8
        Number of bumps in the constraint boundary.
    metric : str, default="score"
        Either "loss" (minimize) or "score" (maximize).
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    n_dim : int
        Number of dimensions (always 2).
    default_bounds : tuple
        Default parameter bounds (-1.25, 1.25).

    Examples
    --------
    >>> from surfaces.test_functions import SimionescuFunction
    >>> func = SimionescuFunction()
    >>> result = func({"x0": 0.84852813, "x1": -0.84852813})
    """

    name = "Simionescu Function"
    _name_ = "simionescu_function"
    __name__ = "SimionescuFunction"

    _spec = {
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
    }

    f_global = -0.072
    x_global = np.array(
        [
            [0.84852813, -0.84852813],
            [-0.84852813, 0.84852813],
        ]
    )

    default_bounds = (-1.25, 1.25)
    n_dim = 2

    latex_formula = (
        r"f(x, y) = 0.1xy \quad \text{s.t.} \quad x^2 + y^2 \le [r_T + r_S\cos(n\arctan(x/y))]^2"
    )
    pgfmath_formula = "0.1*#1*#2"  # Constraint not expressible; use for unconstrained region only

    # Function sheet attributes
    tagline = (
        "A linear surface constrained by a bumpy star-shaped boundary. "
        "Tests handling of complex feasible regions."
    )
    display_bounds = (-1.25, 1.25)
    reference = "Simionescu (2011)"
    reference_url = "https://www.sfu.ca/~ssurjano/simionescu.html"

    def __init__(
        self,
        A=0.1,
        r_T=1,
        r_S=0.2,
        n=8,
        objective="minimize",
        sleep=0,
        memory=False,
        collect_data=True,
        callbacks=None,
        catch_errors=None,
        noise=None,
    ):
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)
        self.n_dim = 2

        self.A = A
        self.r_T = r_T
        self.r_S = r_S
        self.n = n

    def _create_objective_function(self) -> None:
        def simionescu_function(params: Dict[str, Any]) -> float:
            x = np.asarray(params["x0"]).reshape(-1)
            y = np.asarray(params["x1"]).reshape(-1)

            condition = (self.r_T + self.r_S * np.cos(self.n * np.arctan(x / y))) ** 2

            mask = x**2 + y**2 <= condition
            mask_int = mask.astype(int)

            loss = self.A * x * y
            loss = mask_int * loss
            loss[~mask] = np.nan

            # Return scalar if input was scalar
            if loss.size == 1:
                return float(loss[0])
            return loss

        self.pure_objective_function = simionescu_function

    def _search_space(
        self,
        min: float = -1.25,
        max: float = 1.25,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
