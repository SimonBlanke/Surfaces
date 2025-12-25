# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from ..._surrogates import load_surrogate
from .._base_test_function import BaseTestFunction

if TYPE_CHECKING:
    from surfaces.noise import BaseNoise


class MachineLearningFunction(BaseTestFunction):
    """
    Base class for machine learning hyperparameter optimization test functions.

    ML functions evaluate model performance based on hyperparameter configurations.
    They naturally return score values where higher is better.

    Parameters
    ----------
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.
    use_surrogate : bool, default=False
        If True and a pre-trained surrogate exists, use it for fast evaluation.
        Falls back to real evaluation if no surrogate is available.
    """

    _spec = {
        "continuous": False,
        "differentiable": False,
        "stochastic": True,
    }

    para_names: list = []

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space built from *_default class attributes."""
        search_space = {}
        for param_name in self.para_names:
            default_attr = f"{param_name}_default"
            if hasattr(self, default_attr):
                search_space[param_name] = getattr(self, default_attr)
        return search_space

    def __init__(
        self,
        objective: str = "maximize",
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        noise: Optional["BaseNoise"] = None,
        use_surrogate: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(objective, sleep, memory, collect_data, callbacks, catch_errors, noise)
        self.use_surrogate = use_surrogate
        self._surrogate = None

        if use_surrogate:
            self._load_surrogate()

    def _load_surrogate(self) -> None:
        """Load pre-trained surrogate model if available."""
        function_name = getattr(self, "_name_", self.__class__.__name__)
        self._surrogate = load_surrogate(function_name)

        if self._surrogate is None:
            import warnings

            warnings.warn(
                f"No surrogate model found for '{function_name}'. Falling back to real evaluation.",
                UserWarning,
            )
            self.use_surrogate = False

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for surrogate prediction.

        Override in subclasses to add fixed parameters (like dataset, cv)
        that are not in the search space but needed by the surrogate.

        Parameters
        ----------
        params : dict
            Search parameters from the optimizer.

        Returns
        -------
        dict
            Full parameters for surrogate prediction.
        """
        return params

    def _evaluate(self, params: Dict[str, Any]) -> float:
        """Evaluate with timing and objective transformation.

        ML functions naturally return scores (higher is better),
        so we negate when objective is "minimize".
        """
        time.sleep(self.sleep)

        if self.use_surrogate and self._surrogate is not None:
            # Use _get_surrogate_params to include fixed params (dataset, cv)
            surrogate_params = self._get_surrogate_params(params)
            raw_value = self._surrogate.predict(surrogate_params)
        else:
            raw_value = self.pure_objective_function(params)

        if self.objective == "minimize":
            return -raw_value
        return raw_value
