# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for CEC 2010 large-scale benchmark functions."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from .._base_cec import CECFunction


class CEC2010Function(CECFunction):
    """Base class for CEC 2010 large-scale benchmark functions.

    CEC 2010 is a benchmark suite for large-scale global optimization with
    1000 dimensions and partial separability. Variables are divided into
    groups of 50 (m=50, 20 groups total).

    Function categories:
    - F1-F3: Fully separable
    - F4-F18: Partially separable (groups of 50)
    - F19-F20: Non-separable (composition)

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Attributes
    ----------
    func_id : int
        Function ID (1-20).
    n_dim : int
        Number of dimensions (always 1000).
    m : int
        Group size for partial separability (always 50).
    n_groups : int
        Number of groups (always 20).

    References
    ----------
    Tang, K., Li, X., Suganthan, P. N., Yang, Z., & Weise, T. (2010).
    Benchmark Functions for the CEC'2010 Special Session and Competition
    on Large Scale Global Optimization.
    Technical Report, Nature Inspired Computation and Applications Laboratory.
    """

    data_prefix = "cec2010"

    # CEC 2010 only supports 1000 dimensions
    supported_dims: Tuple[int, ...] = (1000,)

    # Partial separability parameters
    m = 50  # Group size
    n_groups = 20  # Number of groups (1000 / 50 = 20)

    _spec = {
        "func_id": None,
        "default_bounds": (-100.0, 100.0),
        "continuous": True,
        "differentiable": True,
        "scalable": False,  # Fixed to 1000D
    }

    # CEC 2010 uses f_global = 0 for all functions
    @property
    def f_global(self) -> float:
        """Global optimum value (always 0 for CEC 2010)."""
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
        # CEC 2010 is fixed to 1000 dimensions
        super().__init__(
            n_dim=1000,
            objective=objective,
            modifiers=modifiers,
            memory=memory,
            collect_data=collect_data,
            callbacks=callbacks,
            catch_errors=catch_errors,
        )

    def _normalize_input(
        self,
        params: Optional[Union[Dict[str, Any], np.ndarray, list, tuple]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert any input format to dict with proper numeric key ordering."""
        if isinstance(params, (np.ndarray, list, tuple)):
            param_names = [f"x{i}" for i in range(self.n_dim)]
            if len(params) != len(param_names):
                raise ValueError(f"Expected {len(param_names)} values, got {len(params)}")
            return {name: params[i] for i, name in enumerate(param_names)}

        if params is None:
            params = {}
        return {**params, **kwargs}

    def _get_permutation(self, index: int = None) -> np.ndarray:
        """Get the permutation indices for this function.

        Parameters
        ----------
        index : int, optional
            Permutation index. If None, uses func_id.

        Returns
        -------
        np.ndarray
            Permutation indices of shape (n_dim,).
        """
        if index is None:
            index = self.func_id
        data = self._load_data()
        key = f"permutation_{index}"
        return data.get(key, np.arange(self.n_dim))

    def _get_group_rotation(self, group_id: int, func_index: int = None) -> np.ndarray:
        """Get the rotation matrix for a specific group.

        Parameters
        ----------
        group_id : int
            Group index (0 to n_groups-1).
        func_index : int, optional
            Function index. If None, uses func_id.

        Returns
        -------
        np.ndarray
            Rotation matrix of shape (m, m).
        """
        if func_index is None:
            func_index = self.func_id
        data = self._load_data()
        key = f"rotation_{func_index}_g{group_id}"
        return data.get(key, np.eye(self.m))

    def _permute(self, x: np.ndarray) -> np.ndarray:
        """Apply permutation to input vector.

        Parameters
        ----------
        x : np.ndarray
            Input vector.

        Returns
        -------
        np.ndarray
            Permuted vector.
        """
        perm = self._get_permutation()
        return x[perm]

    def _get_group(self, z: np.ndarray, group_id: int) -> np.ndarray:
        """Extract a group of variables from permuted vector.

        Parameters
        ----------
        z : np.ndarray
            Permuted and shifted vector.
        group_id : int
            Group index (0 to n_groups-1).

        Returns
        -------
        np.ndarray
            Group variables of shape (m,).
        """
        start = group_id * self.m
        end = start + self.m
        return z[start:end]


class CEC2010SeparableFunction(CEC2010Function):
    """Base class for CEC 2010 fully separable functions (F1-F3)."""

    _spec = {
        **CEC2010Function._spec,
        "separable": True,
    }


class CEC2010PartialSeparableFunction(CEC2010Function):
    """Base class for CEC 2010 partially separable functions (F4-F18).

    Variables are divided into groups of m=50. Variables within a group
    interact, but groups are independent of each other.
    """

    _spec = {
        **CEC2010Function._spec,
        "separable": False,
    }


class CEC2010NonSeparableFunction(CEC2010Function):
    """Base class for CEC 2010 fully non-separable functions (F14-F18)."""

    _spec = {
        **CEC2010Function._spec,
        "separable": False,
    }


class CEC2010CompositionFunction(CEC2010Function):
    """Base class for CEC 2010 composition functions (F19-F20)."""

    _spec = {
        **CEC2010Function._spec,
        "separable": False,
        "unimodal": False,
    }
