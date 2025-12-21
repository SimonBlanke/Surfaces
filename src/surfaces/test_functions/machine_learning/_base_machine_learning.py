# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
from typing import Dict, Any, Optional

from .._base_test_function import BaseTestFunction
from ..._search_data_collection import SearchDataManager, SearchDataCollector


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
    """

    _spec = {
        "continuous": False,
        "differentiable": False,
        "stochastic": True,
    }

    para_names: list = []

    @property
    def default_search_space(self) -> Dict[str, Any]:
        """Default search space built from *_default class attributes."""
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
        evaluate_from_data: bool = False,
        **kwargs
    ):
        super().__init__(objective, sleep)
        self.evaluate_from_data = evaluate_from_data

        if evaluate_from_data:
            self.search_data_manager = SearchDataManager()
            self.data_collector = SearchDataCollector(self.search_data_manager)

    def _evaluate(self, params: Dict[str, Any]) -> float:
        """Evaluate with timing and objective transformation.

        ML functions naturally return scores (higher is better),
        so we negate when objective is "minimize".
        """
        time.sleep(self.sleep)

        if self.evaluate_from_data:
            raw_value = self._objective_function_from_data(params)
        else:
            raw_value = self.pure_objective_function(params)

        if self.objective == "minimize":
            return -raw_value
        return raw_value

    def _objective_function_from_data(self, params: Dict[str, Any]) -> float:
        """Evaluate using stored search data."""
        function_name = getattr(self, '_name_', self.__class__.__name__)

        result = self.search_data_manager.lookup_evaluation(function_name, params)
        if result is None:
            raise ValueError(
                f"No stored evaluation found for parameters: {params}. "
                f"Run data collection first for function: {function_name}"
            )

        score, eval_time = result
        return score

    def _collect_search_data(
        self,
        search_space: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Collect search data for this function."""
        if not hasattr(self, 'data_collector'):
            self.search_data_manager = SearchDataManager()
            self.data_collector = SearchDataCollector(self.search_data_manager)

        return self.data_collector.collect_search_data(
            self, search_space, batch_size, verbose
        )

    def _get_search_data_status(self) -> Dict[str, Any]:
        """Get information about stored search data."""
        if not hasattr(self, 'data_collector'):
            self.search_data_manager = SearchDataManager()
            self.data_collector = SearchDataCollector(self.search_data_manager)

        return self.data_collector.get_collection_status(self)

    def _clear_search_data(self) -> None:
        """Clear all stored search data for this function."""
        if not hasattr(self, 'search_data_manager'):
            self.search_data_manager = SearchDataManager()

        function_name = getattr(self, '_name_', self.__class__.__name__)
        self.search_data_manager.clear_data(function_name)

    def _get_timing_statistics(self) -> Dict[str, float]:
        """Get timing statistics for this function's evaluations."""
        if not hasattr(self, 'data_collector'):
            self.search_data_manager = SearchDataManager()
            self.data_collector = SearchDataCollector(self.search_data_manager)

        return self.data_collector.get_timing_statistics(self)
