# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""0/1 Knapsack test function for binary optimization."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ._base_discrete_function import DiscreteFunction


class KnapsackFunction(DiscreteFunction):
    """0/1 Knapsack problem as a binary test function.

    Each binary variable represents the inclusion (1) or exclusion (0)
    of an item. The objective combines the negated total value with a
    penalty for exceeding the weight capacity, so that feasible
    high-value selections produce the lowest loss.

    .. math::

        f(\\vec{x}) = -\\sum_{i} v_i x_i +
            \\lambda \\cdot \\max\\!\\left(0,\\;
            \\sum_{i} w_i x_i - C\\right)^{2}

    When ``weights`` and ``values`` are not provided, they are generated
    randomly from the given seed. Weights are drawn from
    Uniform[1, 50] (integers) and values from Uniform[1, 100]
    (integers). The default capacity is set to 40% of the total weight,
    creating instances where roughly half the items can fit.

    Parameters
    ----------
    n_items : int, default=20
        Number of items (= number of binary decision variables).
    weights : array-like or None
        Per-item weights. Generated from seed if None.
    values : array-like or None
        Per-item values. Generated from seed if None.
    capacity : float or None
        Knapsack weight capacity. Defaults to 40% of total weight.
    penalty_coefficient : float, default=10.0
        Multiplier for the squared capacity violation.
    seed : int, default=42
        Random seed for generating weights and values.
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Examples
    --------
    >>> func = KnapsackFunction(n_items=5, seed=42)
    >>> # Select all items (may exceed capacity)
    >>> result = func({"x0": 1, "x1": 1, "x2": 1, "x3": 1, "x4": 1})

    References
    ----------
    .. [1] Martello, S., Toth, P. (1990). "Knapsack Problems:
           Algorithms and Computer Implementations." John Wiley & Sons.
    """

    _spec = {
        "eval_cost": 0.1,
        "unimodal": False,
        "separable": False,
        "scalable": True,
        "constrained": True,
    }

    def __init__(
        self,
        n_items: int = 20,
        weights: Optional[Union[list, np.ndarray]] = None,
        values: Optional[Union[list, np.ndarray]] = None,
        capacity: Optional[float] = None,
        penalty_coefficient: float = 10.0,
        seed: int = 42,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        self.seed = seed
        self.penalty_coefficient = penalty_coefficient

        rng = np.random.default_rng(seed)
        if weights is None:
            weights = rng.integers(1, 51, size=n_items)
        if values is None:
            values = rng.integers(1, 101, size=n_items)

        self.weights = np.asarray(weights, dtype=float)
        self.values = np.asarray(values, dtype=float)

        if capacity is None:
            capacity = 0.4 * float(np.sum(self.weights))
        self.capacity = float(capacity)

        if len(self.weights) != n_items or len(self.values) != n_items:
            raise ValueError(
                f"weights and values must have length n_items={n_items}, "
                f"got {len(self.weights)} and {len(self.values)}"
            )

        super().__init__(
            n_items, objective, modifiers, memory, collect_data, callbacks, catch_errors
        )
        self.f_global = None
        self.x_global = None

    def _objective(self, params: Dict[str, Any]) -> float:
        total_value = 0.0
        total_weight = 0.0
        for i in range(self.n_dim):
            x = params[f"x{i}"]
            total_value += self.values[i] * x
            total_weight += self.weights[i] * x

        overweight = max(0.0, total_weight - self.capacity)
        penalty = self.penalty_coefficient * overweight**2
        return -total_value + penalty

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        w = xp.asarray(self.weights)
        v = xp.asarray(self.values)

        total_value = X @ v
        total_weight = X @ w

        overweight = xp.maximum(total_weight - self.capacity, 0.0)
        penalty = self.penalty_coefficient * overweight**2
        return -total_value + penalty
