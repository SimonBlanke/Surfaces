# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for CEC 2019 benchmark functions."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from .._base_cec import CECFunction


class CEC2019Function(CECFunction):
    """Base class for CEC 2019 100-Digit Challenge functions.

    CEC 2019 functions have fixed dimensions per function:
    - F1: Storn's Chebyshev (D=9)
    - F2: Inverse Hilbert (D=16)
    - F3: Lennard-Jones (D=18)
    - F4-F10: Various shifted/rotated (D=10)

    Global optimum f* = 1.0 for all functions.

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Attributes
    ----------
    func_id : int
        Function ID (1-10), set by subclass.
    n_dim : int
        Number of dimensions (fixed per function).
    f_global : float
        Global optimum value (1.0 for all functions).

    References
    ----------
    Price, K. V., Awad, N. H., Ali, M. Z., & Suganthan, P. N. (2018).
    Problem definitions and evaluation criteria for the 100-Digit Challenge
    special session and competition on single objective numerical optimization.
    Technical Report, Nanyang Technological University.
    """

    data_prefix = "cec2019"
    # Each subclass will set its own fixed dimension
    supported_dims: Tuple[int, ...] = ()

    # Fixed dimension for this function (set by subclass)
    _fixed_dim: int = 10

    # CEC 2019 uses f_global = 1.0 for all functions
    @property
    def f_global(self) -> float:
        """Global optimum value (1.0 for all CEC 2019 functions)."""
        return 1.0

    def __init__(
        self,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        # CEC 2019 functions have fixed dimensions
        self.n_dim = self._fixed_dim
        self.supported_dims = (self._fixed_dim,)
        # Call parent with fixed dimension
        super(CECFunction, self).__init__(
            objective, modifiers, memory, collect_data, callbacks, catch_errors
        )

    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load CEC 2019 data (single file for all dimensions)."""
        from .._data_utils import get_data_file

        cache_key = (self.data_prefix, "all")
        if cache_key not in self._data_cache:
            filename = "cec2019_data.npz"
            data_file = get_data_file(self.data_prefix, filename)
            self._data_cache[cache_key] = dict(np.load(data_file))
        return self._data_cache[cache_key]

    def _shift_rotate(self, x):
        """Apply shift then rotation: z = M @ (x - o)."""
        return self._rotate(self._shift(x))
