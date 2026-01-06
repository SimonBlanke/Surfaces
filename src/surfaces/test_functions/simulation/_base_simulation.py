# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Base class for simulation-based test functions.

Simulation functions wrap numerical solvers (FEM, molecular dynamics, CFD, etc.)
to create optimization benchmarks based on real physics and engineering problems.

Unlike algebraic functions which evaluate instantly, simulation functions have
significant evaluation time (seconds to minutes) and may require external
dependencies (FEniCS, OpenMM, Cantera, etc.).
"""

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from surfaces.modifiers import BaseModifier

from .._base_test_function import BaseTestFunction


class SimulationFunction(BaseTestFunction):
    """Base class for simulation-based optimization test functions.

    Simulation functions evaluate objectives by running numerical simulations
    rather than evaluating closed-form expressions. This makes them significantly
    more expensive but also more representative of real-world optimization problems.

    Subclasses should implement:
    - `_setup_simulation()`: Initialize the simulation environment
    - `_run_simulation(params)`: Execute the simulation with given parameters
    - `_extract_objective(result)`: Extract objective value from simulation result

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.
    memory : bool, default=True
        Whether to cache results. Highly recommended for expensive simulations.
    collect_data : bool, default=True
        Whether to collect evaluation history.
    callbacks : callable or list of callable, optional
        Functions called after each evaluation.
    catch_errors : dict, optional
        Mapping of exception types to fallback values.
    timeout : float, optional
        Maximum time in seconds for a single evaluation.
        If exceeded, returns `timeout_value`.
    timeout_value : float, default=inf
        Value to return when simulation times out.

    Attributes
    ----------
    n_dim : int
        Number of input dimensions.
    simulation_time : float
        Estimated time for a single evaluation (seconds).
    requires : list of str
        List of required packages for this simulation.

    Notes
    -----
    Simulation functions benefit greatly from:
    - Memory caching (`memory=True`) to avoid re-running identical simulations
    - Parallel evaluation via `ParallelExecutor` for batch evaluations
    - Surrogate models for expensive optimization campaigns

    Examples
    --------
    >>> from surfaces.test_functions.simulation import TopologyOptimization
    >>> func = TopologyOptimization(mesh_resolution=20)
    >>> result = func({"density_0": 0.5, "density_1": 0.5, ...})
    """

    _spec = {
        "simulation_based": True,
        "expensive": True,
        "continuous": True,
    }

    # Subclasses should override these
    requires: List[str] = []  # Required packages
    estimated_time: float = 1.0  # Estimated seconds per evaluation

    def __init__(
        self,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = True,  # Default True for expensive functions
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        timeout: Optional[float] = None,
        timeout_value: float = float("inf"),
    ) -> None:
        self.timeout = timeout
        self.timeout_value = timeout_value
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)

    def _check_dependencies(self) -> None:
        """Check if required packages are installed."""
        missing = []
        for package in self.requires:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        if missing:
            raise ImportError(
                f"{self.__class__.__name__} requires: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}"
            )

    @abstractmethod
    def _setup_simulation(self) -> None:
        """Initialize the simulation environment.

        Called once during __init__ to set up meshes, load data, etc.
        """
        pass

    @abstractmethod
    def _run_simulation(self, params: Dict[str, Any]) -> Any:
        """Run the simulation with given parameters.

        Parameters
        ----------
        params : dict
            Parameter values for this evaluation.

        Returns
        -------
        Any
            Raw simulation result (to be processed by _extract_objective).
        """
        pass

    @abstractmethod
    def _extract_objective(self, result: Any) -> float:
        """Extract objective value from simulation result.

        Parameters
        ----------
        result : Any
            Raw result from _run_simulation.

        Returns
        -------
        float
            Objective function value.
        """
        pass

    def _create_objective_function(self) -> None:
        """Create the objective function that runs simulations."""
        self._check_dependencies()
        self._setup_simulation()

        def simulation_objective(params: Dict[str, Any]) -> float:
            result = self._run_simulation(params)
            return self._extract_objective(result)

        self.pure_objective_function = simulation_objective
