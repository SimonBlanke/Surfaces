# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Concatenated Trap test function for binary optimization."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ._base_discrete_function import DiscreteFunction


class TrapFunction(DiscreteFunction):
    """Concatenated Trap function (Deb & Goldberg, 1993).

    A deceptive benchmark designed to mislead hillclimbers and
    selection-based optimizers. The bitstring is partitioned into
    blocks of size ``trap_size``. Within each block, the fitness
    gradient points toward all-zeros (the local optimum) while the
    global optimum is at all-ones.

    For a block of size *k* with *u* ones::

        trap(u, k) = k       if u == k    (global optimum)
        trap(u, k) = k-1-u   otherwise    (deceptive slope)

    So a block with 0 ones scores k-1 (deceptive attractor), while a
    block with k-1 ones scores 0 (worst point, right next to optimum).
    This creates a fitness valley that must be crossed to reach the
    global optimum.

    The loss formulation returns ``n_dim - sum(trap values)`` so that
    the minimum is 0 when all bits are 1.

    Parameters
    ----------
    n_dim : int, default=20
        Number of binary variables. Must be divisible by ``trap_size``.
    trap_size : int, default=5
        Size of each trap block.
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Examples
    --------
    >>> func = TrapFunction(n_dim=10, trap_size=5)
    >>> func({"x0": 1, "x1": 1, "x2": 1, "x3": 1, "x4": 1,
    ...       "x5": 1, "x6": 1, "x7": 1, "x8": 1, "x9": 1})
    0
    >>> # All zeros: each block scores trap_size - 1 = 4, total = 8
    >>> func({"x0": 0, "x1": 0, "x2": 0, "x3": 0, "x4": 0,
    ...       "x5": 0, "x6": 0, "x7": 0, "x8": 0, "x9": 0})
    2

    References
    ----------
    .. [1] Deb, K., Goldberg, D.E. (1993). "Analyzing Deception in Trap
           Functions." Foundations of Genetic Algorithms 2, pp. 98-108.
           Morgan Kaufmann.
    .. [2] Ackley, D.H. (1987). "A Connectionist Machine for Genetic
           Hillclimbing." Kluwer Academic Publishers.
    """

    _spec = {
        "eval_cost": 0.1,
        "unimodal": False,
        "separable": False,
        "scalable": True,
        "deceptive": True,
    }

    def __init__(
        self,
        n_dim: int = 20,
        trap_size: int = 5,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        if n_dim % trap_size != 0:
            raise ValueError(
                f"n_dim must be divisible by trap_size, got n_dim={n_dim}, trap_size={trap_size}"
            )
        self.trap_size = trap_size
        super().__init__(n_dim, objective, modifiers, memory, collect_data, callbacks, catch_errors)
        self.f_global = 0.0
        self.x_global = tuple(1 for _ in range(n_dim))

    @staticmethod
    def _trap_value(u: int, k: int) -> int:
        """Compute trap fitness for a block with u ones out of k bits."""
        if u == k:
            return k
        return k - 1 - u

    def _objective(self, params: Dict[str, Any]) -> float:
        total_trap = 0
        for block_start in range(0, self.n_dim, self.trap_size):
            u = 0
            for i in range(block_start, block_start + self.trap_size):
                u += int(params[f"x{i}"])
            total_trap += self._trap_value(u, self.trap_size)
        return self.n_dim - total_trap

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        total_trap = xp.zeros(n_points)
        k = self.trap_size

        for block_start in range(0, self.n_dim, k):
            block = X[:, block_start : block_start + k]
            u = xp.sum(block, axis=1).astype(int)
            # trap(u, k): k if u == k, else k - 1 - u
            trap_vals = np.where(u == k, k, k - 1 - u)
            total_trap = total_trap + trap_vals

        return self.n_dim - total_trap
