# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""LeadingOnes test function for binary optimization."""

from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ._base_discrete_function import DiscreteFunction


class LeadingOnesFunction(DiscreteFunction):
    """LeadingOnes test function.

    Counts the length of the longest prefix of consecutive ones and
    returns the gap to the maximum. The global minimum (loss = 0) is
    reached when the entire bitstring is ones.

    .. math::

        f(\\vec{x}) = n - \\sum_{i=1}^{n} \\prod_{j=1}^{i} x_j

    Unlike OneMax, LeadingOnes is non-separable: the contribution of
    bit i depends on all preceding bits being 1. This creates a strong
    positional dependency that makes the problem harder for optimizers
    that treat variables independently.

    Parameters
    ----------
    n_dim : int, default=20
        Number of binary variables.
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Examples
    --------
    >>> func = LeadingOnesFunction(n_dim=5)
    >>> func({"x0": 1, "x1": 1, "x2": 0, "x3": 1, "x4": 1})
    3
    >>> func({"x0": 1, "x1": 1, "x2": 1, "x3": 1, "x4": 1})
    0

    References
    ----------
    .. [1] Rudolph, G. (1997). "Convergence Properties of Evolutionary
           Algorithms." Verlag Dr. Kovac, Hamburg.
    """

    _spec = {
        "eval_cost": 0.1,
        "unimodal": True,
        "separable": False,
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
        count = 0
        for i in range(self.n_dim):
            if params[f"x{i}"] == 1:
                count += 1
            else:
                break
        return self.n_dim - count

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        # Cumulative product along columns: once a 0 appears, all
        # subsequent products become 0 regardless of later values.
        cumprod = xp.cumprod(X, axis=1)
        leading = xp.sum(cumprod, axis=1)
        return self.n_dim - leading
