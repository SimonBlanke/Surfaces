# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for all CEC competition benchmark functions."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise

from ..algebraic._base_algebraic_function import AlgebraicFunction
from ._data_utils import get_data_file


class CECFunction(AlgebraicFunction):
    """Base class for all CEC competition benchmark functions.

    CEC (IEEE Congress on Evolutionary Computation) benchmark functions are
    shifted and/or rotated versions of classical optimization test functions.
    They are widely used in the optimization research community for comparing
    algorithm performance.

    This base class provides:
    - Data loading (shift vectors, rotation matrices, shuffle indices)
    - Transformation methods (shift, rotate, oscillation, asymmetric)
    - Common interface for all CEC years

    Subclasses must define:
    - func_id: Function identifier within the CEC suite
    - data_prefix: Prefix for data files (e.g., "cec2014")
    - f_global: Global optimum value (via property or formula)
    - supported_dims: Tuple of supported dimensions

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    func_id : int
        Function ID within the CEC suite.
    f_global : float
        Global optimum value.
    x_global : np.ndarray
        Global optimum location (typically the shift vector).
    """

    data_prefix: str = None
    supported_dims: Tuple[int, ...] = ()

    _spec = {
        "func_id": None,
        "default_bounds": (-100.0, 100.0),
        "continuous": True,
        "differentiable": True,
        "scalable": True,
    }

    @property
    def func_id(self) -> Optional[int]:
        """Function ID within the CEC suite."""
        return self.spec.get("func_id")

    # Class-level cache for loaded data, keyed by (data_prefix, n_dim)
    _data_cache: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

    def __init__(
        self,
        n_dim: int = 10,
        objective: str = "minimize",
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        noise: Optional["BaseNoise"] = None,
    ) -> None:
        if n_dim not in self.supported_dims:
            raise ValueError(f"n_dim must be one of {self.supported_dims}, got {n_dim}")
        self.n_dim = n_dim
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)

    @property
    @abstractmethod
    def f_global(self) -> float:
        """Global optimum value for this function."""
        pass

    @property
    def x_global(self) -> Optional[np.ndarray]:
        """Global optimum location (the shift vector)."""
        return self._get_shift_vector()

    @property
    def _data_dir(self) -> Path:
        """Directory containing data files for this CEC year."""
        return Path(__file__).parent / self.data_prefix / "data"

    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load rotation matrices and shift vectors for this dimension.

        Data files are loaded from the surfaces-cec-data package or local
        development directory. Results are cached in memory.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing shift vectors and rotation matrices.

        Raises
        ------
        ImportError
            If surfaces-cec-data is not installed.
        FileNotFoundError
            If the data file is not found.
        """
        cache_key = (self.data_prefix, self.n_dim)
        if cache_key not in self._data_cache:
            filename = f"{self.data_prefix}_data_dim{self.n_dim}.npz"
            data_file = get_data_file(self.data_prefix, filename)
            self._data_cache[cache_key] = dict(np.load(data_file))
        return self._data_cache[cache_key]

    def _get_shift_vector(self, index: int = None) -> np.ndarray:
        """Get the shift vector for this function.

        Parameters
        ----------
        index : int, optional
            Shift vector index. If None, uses func_id.

        Returns
        -------
        np.ndarray
            Shift vector of shape (n_dim,).
        """
        if index is None:
            index = self.func_id
        data = self._load_data()
        key = f"shift_{index}"
        return data.get(key, np.zeros(self.n_dim))

    def _get_rotation_matrix(self, index: int = None) -> np.ndarray:
        """Get a rotation matrix for this function.

        Parameters
        ----------
        index : int, optional
            Rotation matrix index. If None, uses func_id.

        Returns
        -------
        np.ndarray
            Rotation matrix of shape (n_dim, n_dim).
        """
        if index is None:
            index = self.func_id
        data = self._load_data()
        key = f"rotation_{index}"
        return data.get(key, np.eye(self.n_dim))

    def _get_shuffle_indices(self, index: int = None) -> np.ndarray:
        """Get shuffle indices for hybrid functions.

        Parameters
        ----------
        index : int, optional
            Shuffle indices index. If None, uses func_id.

        Returns
        -------
        np.ndarray
            Shuffle indices of shape (n_dim,).
        """
        if index is None:
            index = self.func_id
        data = self._load_data()
        key = f"shuffle_{index}"
        return data.get(key, np.arange(self.n_dim))

    def _shift(self, x: np.ndarray, index: int = None) -> np.ndarray:
        """Apply shift transformation: z = x - o.

        Parameters
        ----------
        x : np.ndarray
            Input vector.
        index : int, optional
            Shift vector index.

        Returns
        -------
        np.ndarray
            Shifted vector.
        """
        return x - self._get_shift_vector(index)

    def _rotate(self, x: np.ndarray, index: int = None) -> np.ndarray:
        """Apply rotation transformation: z = M @ x.

        Parameters
        ----------
        x : np.ndarray
            Input vector.
        index : int, optional
            Rotation matrix index.

        Returns
        -------
        np.ndarray
            Rotated vector.
        """
        return self._get_rotation_matrix(index) @ x

    def _shift_rotate(
        self, x: np.ndarray, shift_index: int = None, rotate_index: int = None
    ) -> np.ndarray:
        """Apply shift then rotation: z = M @ (x - o).

        Parameters
        ----------
        x : np.ndarray
            Input vector.
        shift_index : int, optional
            Shift vector index.
        rotate_index : int, optional
            Rotation matrix index.

        Returns
        -------
        np.ndarray
            Shifted and rotated vector.
        """
        return self._rotate(self._shift(x, shift_index), rotate_index)

    def _oscillation(self, x: np.ndarray) -> np.ndarray:
        """Apply oscillation transformation (T_osz).

        This transformation adds oscillations to the landscape,
        making the function more difficult to optimize.

        Parameters
        ----------
        x : np.ndarray
            Input vector.

        Returns
        -------
        np.ndarray
            Transformed vector.
        """
        z = x.copy()
        for i in range(len(x)):
            if x[i] != 0:
                c1 = 10 if x[i] > 0 else 5.5
                c2 = 7.9 if x[i] > 0 else 3.1
                x_hat = np.log(abs(x[i]))
                z[i] = np.sign(x[i]) * np.exp(
                    x_hat + 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat))
                )
        return z

    def _asymmetric(self, x: np.ndarray, beta: float) -> np.ndarray:
        """Apply asymmetric transformation.

        This transformation makes the landscape asymmetric,
        with different scaling in positive and negative regions.

        Parameters
        ----------
        x : np.ndarray
            Input vector.
        beta : float
            Asymmetry parameter.

        Returns
        -------
        np.ndarray
            Transformed vector.
        """
        z = x.copy()
        D = len(x)
        for i in range(D):
            if x[i] > 0:
                z[i] = x[i] ** (1 + beta * i / (D - 1) * np.sqrt(x[i]))
        return z

    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dict to numpy array.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary with keys "x0", "x1", ..., "x{n_dim-1}".

        Returns
        -------
        np.ndarray
            Array of parameter values.
        """
        return np.array([params[f"x{i}"] for i in range(self.n_dim)])
