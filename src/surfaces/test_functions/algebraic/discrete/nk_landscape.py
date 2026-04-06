# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""NK Landscape test function for binary optimization."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ._base_discrete_function import DiscreteFunction


class NKLandscapeFunction(DiscreteFunction):
    """NK Landscape test function (Kauffman, 1993).

    The standard benchmark for tunable landscape ruggedness in
    evolutionary computation. N binary variables each contribute a
    fitness value that depends on K neighboring variables, creating
    epistatic interactions that control the number of local optima.

    With ``k=0`` the landscape is smooth and unimodal (each variable
    contributes independently). As ``k`` increases toward ``n_dim - 1``,
    the landscape becomes increasingly rugged until it resembles a
    random fitness assignment at ``k = n_dim - 1``.

    The adjacent neighbor model is used: for variable *i*, the K
    neighbors are ``i+1, i+2, ..., i+K`` (wrapping around). Fitness
    contributions are drawn from Uniform[0, 1] using the provided seed.

    .. math::

        F(\\vec{x}) = \\frac{1}{N} \\sum_{i=0}^{N-1} f_i(x_i, x_{i+1}, \\ldots, x_{i+K})

    The loss formulation returns ``1 - F`` so that the minimum is near 0
    for high-fitness solutions.

    Parameters
    ----------
    n_dim : int, default=20
        Number of binary variables (N).
    k : int, default=2
        Number of epistatic neighbors per variable (K).
        Must satisfy 0 <= k < n_dim.
    seed : int, default=42
        Random seed for generating the fitness lookup tables.
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Attributes
    ----------
    fitness_table : list of ndarray
        Pre-computed fitness contributions. ``fitness_table[i]`` has
        shape ``(2^(k+1),)`` and maps each combination of
        ``(x_i, neighbors)`` to a fitness value in [0, 1].

    Examples
    --------
    >>> func = NKLandscapeFunction(n_dim=10, k=2, seed=42)
    >>> result = func({"x0": 1, "x1": 0, "x2": 1, "x3": 0, "x4": 1,
    ...               "x5": 0, "x6": 1, "x7": 0, "x8": 1, "x9": 0})

    References
    ----------
    .. [1] Kauffman, S.A. (1993). "The Origins of Order:
           Self-Organization and Selection in Evolution."
           Oxford University Press.
    .. [2] Kauffman, S.A., Weinberger, E.D. (1989). "The NK model of
           rugged fitness landscapes and its application to maturation
           of the immune response." Journal of Theoretical Biology,
           141(2), 211-245.
    """

    _spec = {
        "eval_cost": 0.2,
        "unimodal": False,
        "separable": False,
        "scalable": True,
    }

    def __init__(
        self,
        n_dim: int = 20,
        k: int = 2,
        seed: int = 42,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        if not 0 <= k < n_dim:
            raise ValueError(f"k must satisfy 0 <= k < n_dim, got k={k}, n_dim={n_dim}")
        self.k = k
        self.seed = seed
        super().__init__(n_dim, objective, modifiers, memory, collect_data, callbacks, catch_errors)
        self.f_global = None
        self.x_global = None
        self._build_fitness_tables()

    def _build_fitness_tables(self) -> None:
        """Generate random fitness lookup tables from the seed."""
        rng = np.random.default_rng(self.seed)
        n_entries = 2 ** (self.k + 1)
        self.fitness_table = [rng.uniform(0.0, 1.0, size=n_entries) for _ in range(self.n_dim)]

    def _get_neighbors(self, i: int) -> list:
        """Return the K adjacent neighbor indices for variable i (circular)."""
        return [(i + j + 1) % self.n_dim for j in range(self.k)]

    def _contribution_index(self, bits: list) -> int:
        """Convert a list of bits to an integer index for the lookup table."""
        idx = 0
        for b in bits:
            idx = (idx << 1) | int(b)
        return idx

    def _objective(self, params: Dict[str, Any]) -> float:
        bits = [params[f"x{i}"] for i in range(self.n_dim)]
        total_fitness = 0.0
        for i in range(self.n_dim):
            group = [bits[i]] + [bits[j] for j in self._get_neighbors(i)]
            idx = self._contribution_index(group)
            total_fitness += self.fitness_table[i][idx]
        mean_fitness = total_fitness / self.n_dim
        return 1.0 - mean_fitness

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        n_points = X.shape[0]
        total_fitness = xp.zeros(n_points)

        for i in range(self.n_dim):
            neighbor_indices = self._get_neighbors(i)
            # Build the lookup index from (x_i, x_{neighbors}) bit pattern
            group_idx = X[:, i].astype(int)
            for j in neighbor_indices:
                group_idx = (group_idx << 1) | X[:, j].astype(int)

            table = xp.asarray(self.fitness_table[i])
            total_fitness = total_fitness + table[group_idx]

        mean_fitness = total_fitness / self.n_dim
        return 1.0 - mean_fitness
