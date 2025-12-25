# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for BBOB (Black-Box Optimization Benchmarking) functions.

The BBOB test suite is part of the COCO (Comparing Continuous Optimizers)
platform and is widely used in the evolutionary computation community.

References:
    Hansen, N., Auger, A., Ros, R., Mersmann, O., Tusar, T., & Brockhoff, D. (2020).
    COCO: A platform for comparing continuous optimizers in a black-box setting.
    arXiv preprint arXiv:1603.08785.

    Finck, S., Hansen, N., Ros, R., & Auger, A. (2009).
    Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions.
    Technical Report 2009/20, Research Center PPE.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

from ..algebraic._base_algebraic_function import AlgebraicFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class BBOBFunction(AlgebraicFunction):
    """Base class for BBOB benchmark functions.

    BBOB functions feature:
    - Search domain of [-5, 5]^D
    - Random optimal location x* within the domain
    - Random optimal value f* from Cauchy distribution (clipped to [-1000, 1000])
    - Transformations: T_osz (oscillation), T_asy (asymmetry), rotations
    - Instance-based: different instances have different random transformations

    Parameters
    ----------
    n_dim : int, default=10
        Number of dimensions. Common values: 2, 3, 5, 10, 20, 40.
    instance : int, default=1
        Instance number (1-15 in standard BBOB). Controls random seed for
        generating optimal location, optimal value, and transformation matrices.
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    func_id : int
        Function ID (1-24), set by subclass.
    x_opt : np.ndarray
        Optimal location in the search space.
    f_opt : float
        Optimal function value.
    """

    _spec = {
        "func_id": None,
        "default_bounds": (-5.0, 5.0),
        "continuous": True,
        "differentiable": True,
        "scalable": True,
    }

    @property
    def func_id(self) -> Optional[int]:
        """Function ID (1-24) within BBOB suite."""
        return self.spec.get("func_id")

    def __init__(
        self,
        n_dim: int = 10,
        instance: int = 1,
        objective: str = "minimize",
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        noise: Optional["BaseNoise"] = None,
    ) -> None:
        self.n_dim = n_dim
        self.instance = instance
        self._rng = np.random.RandomState(self._compute_seed())

        # Generate instance-specific parameters
        self._x_opt = None
        self._f_opt = None
        self._Q = None
        self._R = None
        self._lambda_alpha = None

        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)

    def _compute_seed(self) -> int:
        """Compute random seed from function ID, dimension, and instance."""
        return self.func_id * 10000 + self.n_dim * 100 + self.instance

    @property
    def f_global(self) -> float:
        """Global optimum value for this function instance."""
        return float(self.f_opt)

    @property
    def x_global(self) -> Optional[np.ndarray]:
        """Global optimum location for this function instance."""
        return self.x_opt

    @property
    def x_opt(self) -> np.ndarray:
        """Optimal location in the search space."""
        if self._x_opt is None:
            self._x_opt = self._generate_x_opt()
        return self._x_opt

    @property
    def f_opt(self) -> float:
        """Optimal function value."""
        if self._f_opt is None:
            self._f_opt = self._generate_f_opt()
        return self._f_opt

    def _generate_x_opt(self) -> np.ndarray:
        """Generate optimal location. Override in subclasses if needed."""
        return self._rng.uniform(-4, 4, self.n_dim)

    def _generate_f_opt(self) -> float:
        """Generate optimal value from Cauchy distribution."""
        f = np.round(np.clip(self._rng.standard_cauchy() * 100, -1000, 1000), decimals=2)
        return float(f)

    @property
    def Q(self) -> np.ndarray:
        """First rotation matrix (orthogonal)."""
        if self._Q is None:
            self._Q = self._generate_rotation_matrix()
        return self._Q

    @property
    def R(self) -> np.ndarray:
        """Second rotation matrix (orthogonal)."""
        if self._R is None:
            self._R = self._generate_rotation_matrix()
        return self._R

    def _generate_rotation_matrix(self) -> np.ndarray:
        """Generate a random orthogonal rotation matrix using QR decomposition."""
        A = self._rng.randn(self.n_dim, self.n_dim)
        Q, _ = np.linalg.qr(A)
        return Q

    def lambda_alpha(self, alpha: float) -> np.ndarray:
        """Generate diagonal conditioning matrix with condition number alpha.

        Parameters
        ----------
        alpha : float
            Condition number (ratio of largest to smallest eigenvalue).

        Returns
        -------
        np.ndarray
            Diagonal matrix of shape (n_dim, n_dim).
        """
        i = np.arange(self.n_dim)
        diag = np.power(alpha, 0.5 * i / (self.n_dim - 1)) if self.n_dim > 1 else np.ones(1)
        return np.diag(diag)

    def t_osz(self, x: np.ndarray) -> np.ndarray:
        """Apply oscillation transformation T_osz.

        Creates smooth oscillations around the identity to break symmetry.

        Parameters
        ----------
        x : np.ndarray
            Input vector.

        Returns
        -------
        np.ndarray
            Transformed vector.
        """
        x_hat = np.where(x != 0, np.log(np.abs(x)), 0)
        c1 = np.where(x > 0, 10.0, 5.5)
        c2 = np.where(x > 0, 7.9, 3.1)
        return np.sign(x) * np.exp(x_hat + 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat)))

    def t_asy(self, x: np.ndarray, beta: float) -> np.ndarray:
        """Apply asymmetry transformation T_asy^beta.

        Breaks symmetry by applying different scaling to positive values.

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
        result = x.copy()
        i = np.arange(self.n_dim)
        mask = x > 0
        if self.n_dim > 1:
            exp = 1 + beta * i / (self.n_dim - 1) * np.sqrt(np.maximum(x, 0))
        else:
            exp = 1 + beta * np.sqrt(np.maximum(x, 0))
        result[mask] = np.power(x[mask], exp[mask])
        return result

    def f_pen(self, x: np.ndarray) -> float:
        """Boundary penalty function.

        Penalizes solutions outside [-5, 5]^D.

        Parameters
        ----------
        x : np.ndarray
            Input vector.

        Returns
        -------
        float
            Penalty value (0 if within bounds).
        """
        return np.sum(np.maximum(0, np.abs(x) - 5) ** 2)

    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dict to numpy array."""
        return np.array([params[f"x{i}"] for i in range(self.n_dim)])
