# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""OneMax test function for binary optimization."""

from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ._base_discrete_function import DiscreteFunction


class OneMaxFunction(DiscreteFunction):
    """OneMax (bit-counting) test function.

    The simplest pseudo-boolean benchmark. The objective counts how many
    bits differ from 1, so the global minimum is reached when all
    variables are set to 1.

    .. math::

        f(\\vec{x}) = n - \\sum_{i=1}^{n} x_i

    The function is separable, unimodal, and serves the same role in
    discrete optimization that the Sphere function serves in continuous
    optimization: a basic sanity check.

    Parameters
    ----------
    n_dim : int, default=20
        Number of binary variables.
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Examples
    --------
    >>> func = OneMaxFunction(n_dim=5)
    >>> func({"x0": 1, "x1": 1, "x2": 1, "x3": 1, "x4": 1})
    0
    >>> func({"x0": 0, "x1": 0, "x2": 0, "x3": 0, "x4": 0})
    5

    References
    ----------
    .. [1] Ackley, D.H. (1987). "A Connectionist Machine for Genetic
           Hillclimbing." Kluwer Academic Publishers.
    """

    _spec = {
        "eval_cost": 0.1,
        "unimodal": True,
        "separable": True,
        "scalable": True,
    }

    def __init__(
        self,
        n_dim: int = 20,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        super().__init__(n_dim, objective, modifiers, memory, collect_data, callbacks, catch_errors)
        self.f_global = 0.0
        self.x_global = tuple(1 for _ in range(n_dim))

    def _objective(self, params: Dict[str, Any]) -> float:
        total = 0
        for i in range(self.n_dim):
            total += params[f"x{i}"]
        return self.n_dim - total

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        return self.n_dim - xp.sum(X, axis=1)
