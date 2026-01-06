# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Custom test function for user-defined objectives.

This module is private and not exported in __init__.py.
The API may change in future versions.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from ._base_test_function import BaseTestFunction


class CustomTestFunction(BaseTestFunction):
    """User-defined test function with full Surfaces infrastructure.

    Allows users to wrap any callable as a Surfaces test function,
    gaining access to all features: modifiers, memory, callbacks,
    batch evaluation, etc.

    Parameters
    ----------
    objective_fn : callable
        The objective function to evaluate. Must accept a dict of parameters
        and return a float. Signature: ``fn(params: dict) -> float``
    search_space : dict
        Search space definition. Can be:
        - Dict mapping param names to arrays: ``{"x": np.linspace(-5, 5, 100)}``
        - Dict mapping param names to bounds tuples: ``{"x": (-5, 5)}``
          (will be converted to arrays using `resolution`)
    resolution : int, default=100
        Number of points per dimension when bounds tuples are provided.
    global_optimum : dict, optional
        Known global optimum information with keys:
        - "position": dict mapping param names to optimal values
        - "score": the optimal score value
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.
    memory : bool, default=False
        If True, caches evaluated positions.
    collect_data : bool, default=True
        If True, collects evaluation data.
    callbacks : callable or list of callables, optional
        Function(s) called after each evaluation.
    catch_errors : dict, optional
        Dictionary mapping exception types to return values.

    Examples
    --------
    Basic usage with a simple function:

    >>> def my_sphere(params):
    ...     return sum(v**2 for v in params.values())
    >>> func = CustomTestFunction(
    ...     objective_fn=my_sphere,
    ...     search_space={"x": (-5, 5), "y": (-5, 5)},
    ... )
    >>> func(x=1, y=2)
    5

    With explicit search space arrays:

    >>> func = CustomTestFunction(
    ...     objective_fn=my_sphere,
    ...     search_space={
    ...         "x": np.linspace(-5, 5, 50),
    ...         "y": np.linspace(-10, 10, 100),
    ...     },
    ... )

    With global optimum and modifiers:

    >>> from surfaces.modifiers import GaussianNoise
    >>> func = CustomTestFunction(
    ...     objective_fn=my_sphere,
    ...     search_space={"x": (-5, 5), "y": (-5, 5)},
    ...     global_optimum={"position": {"x": 0, "y": 0}, "score": 0},
    ...     modifiers=[GaussianNoise(sigma=0.1)],
    ... )
    >>> func.f_global
    0
    >>> func.x_global
    {'x': 0, 'y': 0}

    Notes
    -----
    This class is currently private (not exported in __init__.py).
    The API may change in future versions.
    """

    _spec = {
        "name": "CustomTestFunction",
        "continuous": None,  # Unknown for user functions
        "differentiable": None,
        "convex": None,
        "separable": None,
        "unimodal": None,
        "scalable": None,
    }

    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: Dict[str, Union[np.ndarray, Tuple[float, float], List[float]]],
        resolution: int = 100,
        global_optimum: Optional[Dict[str, Any]] = None,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        self._user_objective_fn = objective_fn
        self._search_space = self._normalize_search_space(search_space, resolution)
        self._resolution = resolution

        # Set global optimum if provided
        if global_optimum is not None:
            self.x_global = global_optimum.get("position")
            self.f_global = global_optimum.get("score")

        # Update spec with n_dim
        self._spec = {**self._spec, "n_dim": len(self._search_space)}

        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

    def _normalize_search_space(
        self,
        search_space: Dict[str, Union[np.ndarray, Tuple[float, float], List[float]]],
        resolution: int,
    ) -> Dict[str, np.ndarray]:
        """Convert search space to dict of arrays."""
        normalized = {}
        for name, value in search_space.items():
            if isinstance(value, np.ndarray):
                normalized[name] = value
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                # Bounds tuple: (min, max)
                min_val, max_val = value
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    normalized[name] = np.linspace(min_val, max_val, resolution)
                else:
                    # It's a list of values, not bounds
                    normalized[name] = np.array(value)
            elif isinstance(value, list):
                normalized[name] = np.array(value)
            else:
                raise ValueError(
                    f"Invalid search space for '{name}': expected array, list, or "
                    f"(min, max) tuple, got {type(value).__name__}"
                )
        return normalized

    def _create_objective_function(self) -> None:
        """Set the pure objective function from user-provided callable."""
        self.pure_objective_function = self._user_objective_fn

    @property
    def search_space(self) -> Dict[str, np.ndarray]:
        """Search space for this function."""
        return self._search_space

    @property
    def n_dim(self) -> int:
        """Number of dimensions."""
        return len(self._search_space)

    @property
    def param_names(self) -> List[str]:
        """Parameter names in sorted order."""
        return sorted(self._search_space.keys())

    def __repr__(self) -> str:
        return (
            f"CustomTestFunction(n_dim={self.n_dim}, "
            f"params={self.param_names}, "
            f"objective='{self.objective}')"
        )
