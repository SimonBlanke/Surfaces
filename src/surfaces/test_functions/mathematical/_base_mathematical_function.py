# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

from .._base_test_function import BaseTestFunction


class MathematicalFunction(BaseTestFunction):
    """Base class for mathematical optimization test functions.

    Mathematical functions compute a loss value based on input parameters.
    The loss can be transformed to a score (negated) via the metric parameter
    or by using the explicit loss() and score() methods.

    Parameters
    ----------
    metric : str, default="loss"
        Either "loss" (minimize) or "score" (maximize).
        Controls the return value of __call__().
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.
    validate : bool, default=True
        Whether to validate parameters against the search space.

    Examples
    --------
    >>> from surfaces.test_functions import SphereFunction
    >>> func = SphereFunction(n_dim=2)
    >>> search_space = func.default_search_space
    >>> result = func({"x0": 0.0, "x1": 0.0})
    """

    # Default bounds for mathematical functions (can be overridden by subclasses)
    default_bounds: Tuple[float, float] = (-5.0, 5.0)
    default_size: int = 10000

    @property
    def default_search_space(self) -> Dict[str, Any]:
        """
        Default search space for this function.

        Returns a dictionary mapping parameter names to numpy arrays of values
        based on the function's default_bounds and n_dim.
        """
        min_val, max_val = self.default_bounds
        return self._create_search_space(min=min_val, max=max_val, size=self.default_size)

    def _create_search_space(
        self,
        min: Union[float, list] = -5,
        max: Union[float, list] = 5,
        size: int = 10000,
        value_types: str = "array"
    ) -> Dict[str, Any]:
        """
        Create a search space with custom bounds.

        Internal method for creating search spaces with non-default parameters.
        For most use cases, access default_search_space property instead.
        """
        return self._create_n_dim_search_space(min=min, max=max, size=size, value_types=value_types)

    def __init__(
        self,
        metric: str = "loss",
        sleep: float = 0,
        validate: bool = True,
    ):
        """
        Initialize a mathematical test function.

        Args:
            metric: Either "loss" (minimize, default) or "score" (maximize).
                   Controls the return value of objective_function() and __call__().
                   For explicit control, use loss() or score() methods instead.
            sleep: Artificial delay in seconds added to each evaluation
            validate: Whether to validate parameters against search space bounds
        """
        super().__init__(metric, sleep, validate)

        self.metric = metric
        self.sleep = sleep

    def _return_metric(self, loss: float) -> float:
        """
        Transform raw loss value based on metric setting.

        This maintains backward compatibility with existing code.
        For new code, prefer using loss() or score() methods explicitly.
        """
        if self.metric == "score":
            return -loss
        elif self.metric == "loss":
            return loss
        else:
            raise ValueError(f"Invalid metric: {self.metric}. Must be 'loss' or 'score'.")

    def _to_loss(self, raw_value: float) -> float:
        """
        Convert raw value to loss (for minimization).

        Mathematical functions naturally return loss values,
        so this is an identity transformation.
        """
        return raw_value

    def _to_score(self, raw_value: float) -> float:
        """
        Convert raw value to score (for maximization).

        Mathematical functions naturally return loss values,
        so the score is the negated loss.
        """
        return -raw_value

    @staticmethod
    def _conv_arrays2lists(search_space: Dict[str, Any]) -> Dict[str, list]:
        """Convert array-valued search space to list-valued."""
        return {
            para_name: list(dim_values)
            for para_name, dim_values in search_space.items()
        }

    def _create_n_dim_search_space(
        self,
        min: Union[float, list] = -5,
        max: Union[float, list] = 5,
        size: int = 100,
        value_types: str = "array"
    ) -> Dict[str, Any]:
        """
        Create a search space for an N-dimensional function.

        Args:
            min: Lower bound(s). Either a single value for all dimensions
                 or a list of per-dimension bounds.
            max: Upper bound(s). Either a single value for all dimensions
                 or a list of per-dimension bounds.
            size: Total number of grid points across all dimensions
            value_types: "array" for numpy arrays, "list" for Python lists

        Returns:
            Dictionary mapping parameter names ('x0', 'x1', ...) to value arrays/lists
        """
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
