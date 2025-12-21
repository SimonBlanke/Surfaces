# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for CEC 2014 benchmark functions."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from ...mathematical._base_mathematical_function import MathematicalFunction


class CEC2014Function(MathematicalFunction):
    """Base class for CEC 2014 benchmark functions.

    CEC 2014 functions are shifted and/or rotated versions of classical
    optimization test functions. Each function has:
    - A function ID (1-30)
    - A global optimum value f* = func_id * 100
    - Search bounds of [-100, 100]^D
    - Support for dimensions: 10, 20, 30, 50, 100

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Supported: 10, 20, 30, 50, 100.
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    func_id : int
        Function ID (1-30), set by subclass.
    f_global : float
        Global optimum value (func_id * 100).

    References
    ----------
    Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013).
    Problem definitions and evaluation criteria for the CEC 2014 special
    session and competition on single objective real-parameter numerical
    optimization.
    """

    func_id: int = None
    default_bounds: Tuple[float, float] = (-100.0, 100.0)
    supported_dims: Tuple[int, ...] = (10, 20, 30, 50, 100)

    _spec = {
        "continuous": True,
        "differentiable": True,
        "scalable": True,
    }

    # Class-level cache for loaded data
    _data_cache: Dict[int, Dict[str, np.ndarray]] = {}
    _data_dir: Path = Path(__file__).parent / "data"

    def __init__(
        self,
        n_dim: int = 10,
        objective: str = "minimize",
        sleep: float = 0,
    ):
        if n_dim not in self.supported_dims:
            raise ValueError(
                f"n_dim must be one of {self.supported_dims}, got {n_dim}"
            )
        self.n_dim = n_dim
        super().__init__(objective, sleep)

    @property
    def f_global(self) -> float:
        """Global optimum value for this function."""
        return float(self.func_id * 100)

    @property
    def x_global(self) -> Optional[np.ndarray]:
        """Global optimum location (the shift vector)."""
        return self._get_shift_vector()

    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load rotation matrices and shift vectors for this dimension."""
        if self.n_dim not in self._data_cache:
            data_file = self._data_dir / f"cec2014_data_dim{self.n_dim}.npz"
            if not data_file.exists():
                raise FileNotFoundError(
                    f"CEC 2014 data file not found: {data_file}\n"
                    f"Download the official CEC 2014 data files from:\n"
                    f"https://github.com/P-N-Suganthan/CEC2014\n"
                    f"Then convert them using: surfaces.tools.convert_cec2014_data()"
                )
            self._data_cache[self.n_dim] = dict(np.load(data_file))
        return self._data_cache[self.n_dim]

    def _get_shift_vector(self) -> np.ndarray:
        """Get the shift vector for this function."""
        data = self._load_data()
        key = f"shift_{self.func_id}"
        return data.get(key, np.zeros(self.n_dim))

    def _get_rotation_matrix(self) -> np.ndarray:
        """Get the rotation matrix for this function."""
        data = self._load_data()
        key = f"rotation_{self.func_id}"
        return data.get(key, np.eye(self.n_dim))

    def _get_shuffle_indices(self) -> np.ndarray:
        """Get shuffle indices for hybrid functions."""
        data = self._load_data()
        key = f"shuffle_{self.func_id}"
        return data.get(key, np.arange(self.n_dim))

    def _shift(self, x: np.ndarray) -> np.ndarray:
        """Apply shift transformation: z = x - o."""
        return x - self._get_shift_vector()

    def _rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply rotation transformation: z = M @ x."""
        return self._get_rotation_matrix() @ x

    def _shift_rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply shift then rotation: z = M @ (x - o)."""
        return self._rotate(self._shift(x))

    def _asymmetric(self, x: np.ndarray, beta: float) -> np.ndarray:
        """Apply asymmetric transformation."""
        z = x.copy()
        D = len(x)
        for i in range(D):
            if x[i] > 0:
                z[i] = x[i] ** (1 + beta * i / (D - 1) * np.sqrt(x[i]))
        return z

    def _oscillation(self, x: np.ndarray) -> np.ndarray:
        """Apply oscillation transformation (T_osz)."""
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

    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dict to numpy array."""
        return np.array([params[f"x{i}"] for i in range(self.n_dim)])
