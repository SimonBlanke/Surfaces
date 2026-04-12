# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for discrete (pseudo-boolean/combinatorial) test functions."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from .._base_algebraic_function import AlgebraicFunction


class DiscreteFunction(AlgebraicFunction):
    """Base class for discrete optimization test functions.

    Discrete functions operate on binary or integer search spaces rather
    than continuous ones. The default search space assigns each variable
    the values ``[0, 1]``, making these functions suitable for benchmarking
    combinatorial and pseudo-boolean optimizers.

    Subclasses implement ``_objective(params)`` to define the fitness
    landscape. By convention, ``_objective`` returns a loss (lower is
    better), consistent with the rest of the algebraic function family.

    Parameters
    ----------
    n_dim : int, default=20
        Number of binary decision variables.
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> func = OneMaxFunction(n_dim=10)
    >>> func({"x0": 1, "x1": 1, "x2": 0, "x3": 1, "x4": 0,
    ...       "x5": 1, "x6": 1, "x7": 0, "x8": 1, "x9": 1})
    3
    """

    _spec = {
        "continuous": False,
        "differentiable": False,
        "convex": False,
        "discrete": True,
        "default_bounds": (0, 1),
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
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)
        self.n_dim = n_dim

    def _default_search_space(self) -> Dict[str, Any]:
        """Binary search space: each variable takes values 0 or 1."""
        return {f"x{i}": np.array([0, 1]) for i in range(self.n_dim)}
