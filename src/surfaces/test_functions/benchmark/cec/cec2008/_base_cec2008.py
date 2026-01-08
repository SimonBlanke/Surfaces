# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for CEC 2008 large-scale benchmark functions."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from .._base_cec import CECFunction


class CEC2008Function(CECFunction):
    """Base class for CEC 2008 large-scale benchmark functions.

    CEC 2008 is a benchmark suite for large-scale global optimization with
    1000 dimensions. The functions include:
    - Separable functions: F1, F2, F4, F6
    - Non-separable functions: F3, F5, F7

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Attributes
    ----------
    func_id : int
        Function ID (1-7).
    n_dim : int
        Number of dimensions (always 1000).
    f_global : float
        Global optimum value (always 0 for CEC 2008).

    References
    ----------
    Tang, K., Li, X., Suganthan, P. N., Yang, Z., & Weise, T. (2008).
    Benchmark Functions for the CEC'2008 Special Session and Competition
    on Large Scale Global Optimization.
    Technical Report, Nature Inspired Computation and Applications Laboratory.
    """

    data_prefix = "cec2008"

    # CEC 2008 only supports 1000 dimensions
    supported_dims: Tuple[int, ...] = (1000,)

    def _normalize_input(
        self,
        params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert any input format to dict with proper numeric key ordering.

        Overrides base class to fix alphabetical sorting issue for
        high-dimensional functions (x10 sorted before x2 alphabetically).
        """
        if isinstance(params, (np.ndarray, list, tuple)):
            # Use numeric ordering for keys (x0, x1, ..., x999)
            param_names = [f"x{i}" for i in range(self.n_dim)]
            if len(params) != len(param_names):
                raise ValueError(f"Expected {len(param_names)} values, got {len(params)}")
            return {name: params[i] for i, name in enumerate(param_names)}

        if params is None:
            params = {}
        return {**params, **kwargs}

    _spec = {
        "func_id": None,
        "default_bounds": (-100.0, 100.0),
        "continuous": True,
        "differentiable": True,
        "scalable": False,  # Fixed to 1000D
    }

    # CEC 2008 uses f_global = 0 for all functions
    @property
    def f_global(self) -> float:
        """Global optimum value (always 0 for CEC 2008)."""
        return 0.0

    def __init__(
        self,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        # CEC 2008 is fixed to 1000 dimensions
        super().__init__(
            n_dim=1000,
            objective=objective,
            modifiers=modifiers,
            memory=memory,
            collect_data=collect_data,
            callbacks=callbacks,
            catch_errors=catch_errors,
        )


class CEC2008SeparableFunction(CEC2008Function):
    """Base class for CEC 2008 separable functions.

    Separable functions (F1, F2, F4, F6) don't use rotation matrices.
    Only shift vectors are applied.
    """

    _spec = {
        **CEC2008Function._spec,
        "separable": True,
    }

    def _get_rotation_matrix(self, index: int = None) -> np.ndarray:
        """Return identity matrix (no rotation for separable functions)."""
        return np.eye(self.n_dim)


class CEC2008NonSeparableFunction(CEC2008Function):
    """Base class for CEC 2008 non-separable functions.

    Non-separable functions (F3, F5, F7) use rotation matrices.
    """

    _spec = {
        **CEC2008Function._spec,
        "separable": False,
    }
