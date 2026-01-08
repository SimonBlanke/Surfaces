# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for multi-objective optimization test functions."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from surfaces._array_utils import ArrayLike, is_array_like
from surfaces.modifiers import BaseModifier


class MultiObjectiveFunction:
    """Base class for multi-objective test functions.

    Multi-objective functions return a vector of objective values instead of
    a scalar. The goal is typically to find the Pareto front - the set of
    solutions where no objective can be improved without worsening another.

    Parameters
    ----------
    n_dim : int
        Number of input dimensions.
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.
        Note: Value-modifying modifiers (like noise) are not supported for
        multi-objective functions. Only side-effect modifiers (like delay)
        work correctly.

    Attributes
    ----------
    n_objectives : int
        Number of objectives (set by subclass).
    n_dim : int
        Number of input dimensions.

    Examples
    --------
    >>> func = ZDT1(n_dim=30)
    >>> result = func(np.zeros(30))
    >>> result.shape
    (2,)
    """

    pure_objective_function: callable
    n_objectives: int = 2
    default_bounds: Tuple[float, float] = (0.0, 1.0)
    default_size: int = 1000

    _spec: Dict[str, Any] = {
        "continuous": True,
        "differentiable": True,
        "convex": False,
        "scalable": True,
    }

    @property
    def spec(self) -> Dict[str, Any]:
        """Function characteristics merged from class hierarchy (read-only)."""
        result = {}
        for klass in reversed(type(self).__mro__):
            if hasattr(klass, "_spec"):
                result.update(klass._spec)
        return result

    def _create_objective_function_(func):
        """Decorator that calls _create_objective_function after __init__."""

        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self._create_objective_function()

        return wrapper

    @_create_objective_function_
    def __init__(self, n_dim: int, modifiers: Optional[List[BaseModifier]] = None):
        self.n_dim = n_dim
        self._modifiers: List[BaseModifier] = modifiers if modifiers is not None else []

    def _create_objective_function(self):
        raise NotImplementedError("'_create_objective_function' must be implemented")

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space based on default_bounds and n_dim."""
        min_val, max_val = self.default_bounds
        return self._create_search_space(min=min_val, max=max_val, size=self.default_size)

    def _create_search_space(
        self,
        min: float = 0.0,
        max: float = 1.0,
        size: int = 1000,
    ) -> Dict[str, Any]:
        """Create search space for the function."""
        search_space = {}
        dim_size = int(size ** (1 / self.n_dim))

        for dim in range(self.n_dim):
            dim_str = f"x{dim}"
            step_size = (max - min) / dim_size
            values = np.arange(min, max, step_size)
            search_space[dim_str] = values

        return search_space

    def __call__(
        self, params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None, **kwargs
    ) -> np.ndarray:
        """Evaluate the multi-objective function.

        Parameters
        ----------
        params : dict, array, list, or tuple
            Parameter values.

        Returns
        -------
        np.ndarray
            Vector of objective values with shape (n_objectives,).

        Examples
        --------
        >>> func({"x0": 0.5, "x1": 0.5})     # dict
        >>> func(np.array([0.5, 0.5]))       # array
        >>> func([0.5, 0.5])                 # list
        """
        params = self._normalize_input(params, **kwargs)
        return self._evaluate(params)

    def _normalize_input(
        self, params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Convert any input format to dict."""
        if isinstance(params, (np.ndarray, list, tuple)):
            param_names = sorted(self.search_space.keys())
            if len(params) != len(param_names):
                raise ValueError(f"Expected {len(param_names)} values, got {len(params)}")
            return {name: params[i] for i, name in enumerate(param_names)}

        if params is None:
            params = {}
        return {**params, **kwargs}

    def _evaluate(self, params: Dict[str, Any]) -> np.ndarray:
        """Evaluate with modifiers.

        Note: Modifiers are applied to the first objective value only,
        primarily for side-effects like delays. Value-modifying modifiers
        (like noise) are not recommended for multi-objective functions.
        """
        result = self.pure_objective_function(params)

        # Apply modifiers for side-effects (e.g., delay)
        # We pass the first objective value through the modifier pipeline
        if self._modifiers:
            context = {}
            value = result[0]
            for modifier in self._modifiers:
                value = modifier.apply(value, params, context)

        return result

    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dict to numpy array."""
        return np.array([params[f"x{i}"] for i in range(self.n_dim)])

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        """Generate points on the theoretical Pareto front.

        Parameters
        ----------
        n_points : int, default=100
            Number of points to generate on the Pareto front.

        Returns
        -------
        np.ndarray
            Array of shape (n_points, n_objectives) containing points
            on the Pareto front.
        """
        raise NotImplementedError("'pareto_front' must be implemented by subclass")

    def pareto_set(self, n_points: int = 100) -> np.ndarray:
        """Generate points in the Pareto set (decision space).

        Parameters
        ----------
        n_points : int, default=100
            Number of points to generate in the Pareto set.

        Returns
        -------
        np.ndarray
            Array of shape (n_points, n_dim) containing points
            in the Pareto set.
        """
        raise NotImplementedError("'pareto_set' must be implemented by subclass")

    # =========================================================================
    # Batch Evaluation
    # =========================================================================

    def batch(self, X: ArrayLike) -> ArrayLike:
        """Evaluate multiple parameter sets in a single call.

        This method enables efficient batch evaluation through vectorization.
        The input array type determines the computation backend (numpy, cupy, jax).

        Parameters
        ----------
        X : ArrayLike
            2D array of shape (n_points, n_dim) where each row is a parameter set.
            Supports numpy, cupy, and jax arrays. The output array type matches
            the input type.

        Returns
        -------
        ArrayLike
            2D array of shape (n_points, n_objectives) with evaluation results.
            Each row contains objective values for one parameter set.
            Returns the same array type as input (numpy -> numpy, cupy -> cupy).

        Raises
        ------
        NotImplementedError
            If the function does not implement _batch_objective.
        ValueError
            If X has wrong number of dimensions or wrong n_dim.

        Examples
        --------
        >>> import numpy as np
        >>> func = ZDT1(n_dim=30)
        >>> X = np.random.rand(100, 30)  # 100 points in 30D
        >>> results = func.batch(X)
        >>> results.shape
        (100, 2)  # 100 points, 2 objectives
        """
        if not hasattr(self, "_batch_objective"):
            raise NotImplementedError(
                f"{type(self).__name__} does not support vectorized batch evaluation. "
                "Implement _batch_objective(X) to enable this feature."
            )

        # Validate input
        if not is_array_like(X):
            raise TypeError(
                f"Expected array-like input with shape (n_points, n_dim), "
                f"got {type(X).__name__}"
            )

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array with shape (n_points, n_dim), got {X.ndim}D array")

        if X.shape[1] != self.n_dim:
            raise ValueError(f"Expected {self.n_dim} dimensions, got {X.shape[1]}")

        # Compute vectorized result
        return self._batch_objective(X)
