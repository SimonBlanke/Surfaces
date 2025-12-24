# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for algebraic test functions with closed-form expressions."""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise

from .._base_test_function import BaseTestFunction


class AlgebraicFunction(BaseTestFunction):
    """Base class for algebraic optimization test functions.

    Algebraic functions are defined by closed-form analytical expressions,
    as opposed to data-driven (ML) or externally-defined (CEC/BBOB) functions.

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Examples
    --------
    >>> func = SphereFunction(n_dim=2)
    >>> func({"x0": 0.0, "x1": 0.0})
    >>> func(np.array([0.0, 0.0]))
    """

    _spec = {
        "default_bounds": (-5.0, 5.0),
        "continuous": True,
        "differentiable": True,
    }

    default_size: int = 10000

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space based on default_bounds and n_dim."""
        min_val, max_val = self.default_bounds
        return self._create_n_dim_search_space(min=min_val, max=max_val, size=self.default_size)

    def __init__(
        self,
        objective: str = "minimize",
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        noise: Optional["BaseNoise"] = None,
    ) -> None:
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)

    def _create_n_dim_search_space(
        self,
        min: Union[float, List[float]] = -5,
        max: Union[float, List[float]] = 5,
        size: int = 100,
        value_types: str = "array",
    ) -> Dict[str, Any]:
        """Create search space for N-dimensional function."""
        search_space_ = {}
        dim_size = size ** (1 / self.n_dim)

        def add_dim(search_space_: dict, dim: int, min_val, max_val):
            dim_str = "x" + str(dim)
            step_size = (max_val - min_val) / dim_size
            values = np.arange(min_val, max_val, step_size)
            if value_types == "list":
                values = list(values)
            search_space_[dim_str] = values

        if isinstance(min, list) and isinstance(max, list):
            if len(min) != len(max) or len(min) != self.n_dim:
                raise ValueError(
                    f"min and max lists must have length {self.n_dim}, "
                    f"got {len(min)} and {len(max)}"
                )
            for dim, (min_, max_) in enumerate(zip(min, max)):
                add_dim(search_space_, dim, min_, max_)
        else:
            for dim in range(self.n_dim):
                add_dim(search_space_, dim, min, max)

        return search_space_


# Backwards compatibility alias
MathematicalFunction = AlgebraicFunction
