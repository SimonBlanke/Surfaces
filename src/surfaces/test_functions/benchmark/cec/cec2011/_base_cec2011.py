# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for CEC 2011 real-world benchmark functions."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from ....algebraic._base_algebraic_function import AlgebraicFunction


class CEC2011Function(AlgebraicFunction):
    """Base class for CEC 2011 real-world optimization problems.

    CEC 2011 consists of 22 real-world optimization problems with fixed
    dimensions per problem. This implementation covers P01-P08, which are
    analytically defined (no external data/simulators needed).

    Problems P01-P08:
    - P01: FM Sound Synthesis Parameter Estimation (D=6)
    - P02: Lennard-Jones Minimum Energy Cluster (D=30, 10 atoms)
    - P03: Bifunctional Catalyst Blend Optimization (D=1)
    - P04: Stirred Tank Reactor (D=1)
    - P05: Tersoff Potential Si-B (D=30)
    - P06: Tersoff Potential Si-C (D=30)
    - P07: Radar Polyphase Code Design (D=20)
    - P08: Spread Spectrum Radar Code Design (D=7)

    Note: P09-P13 are NOT implemented due to GPL-3.0 license restrictions.

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".

    Attributes
    ----------
    problem_id : int
        Problem ID (1-8).
    n_dim : int
        Number of dimensions (fixed per problem).
    f_global : float
        Best known objective value.
    x_global : np.ndarray
        Best known solution location.

    References
    ----------
    Das, S. & Suganthan, P. N. (2010). Problem Definitions and Evaluation
    Criteria for CEC 2011 Competition on Testing Evolutionary Algorithms
    on Real World Optimization Problems. Technical Report.
    """

    # Each subclass sets its own fixed dimension
    _fixed_dim: int = 1
    _problem_id: int = 0

    _spec = {
        "problem_id": None,
        "default_bounds": (-100.0, 100.0),
        "continuous": True,
        "differentiable": True,
        "scalable": False,  # Fixed dimensions
    }

    @property
    def problem_id(self) -> Optional[int]:
        """Problem ID within the CEC 2011 suite."""
        return self._spec.get("problem_id", self._problem_id)

    @property
    def f_global(self) -> float:
        """Best known objective value."""
        return self._f_global

    @property
    def x_global(self) -> Optional[np.ndarray]:
        """Best known solution location."""
        return self._x_global

    def __init__(
        self,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        # CEC 2011 functions have fixed dimensions
        self.n_dim = self._fixed_dim
        # Initialize best known values (can be overridden by subclass)
        self._f_global = 0.0
        self._x_global = np.zeros(self.n_dim)

        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

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
