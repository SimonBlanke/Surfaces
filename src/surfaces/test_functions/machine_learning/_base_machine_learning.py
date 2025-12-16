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

    ML functions evaluate model performance (e.g., accuracy, R2 score) based on
    hyperparameter configurations. They naturally return score values where
    higher is better.
    """

    # Subclasses should define para_names and corresponding *_default attributes
    para_names: list = []

    @property
    def default_search_space(self) -> Dict[str, Any]:
        """
        Default search space for this ML function.

        Returns a dictionary mapping parameter names to lists of values,
        built from the *_default class attributes.
        """
        search_space = {}
        for param_name in self.para_names:
            default_attr = f"{param_name}_default"
            if hasattr(self, default_attr):
                search_space[param_name] = getattr(self, default_attr)
        return search_space

    def __init__(
        self,
        metric: str = "score",
        sleep: float = 0,
        validate: bool = True,
        evaluate_from_data: bool = False,
        **kwargs
    ):
        """
        Initialize a machine learning test function.

        Args:
            metric: Either "score" (maximize, default) or "loss" (minimize).
                   Controls the return value of objective_function() and __call__().
                   For explicit control, use loss() or score() methods instead.
            sleep: Artificial delay in seconds added to each evaluation
            validate: Whether to validate parameters against search space
            evaluate_from_data: If True, use pre-computed search data for fast lookups
        """
        super().__init__(metric, sleep, validate)

        self.evaluate_from_data = evaluate_from_data

        if evaluate_from_data:
            self.search_data_manager = SearchDataManager()
            self.data_collector = SearchDataCollector(self.search_data_manager)

    def _get_raw_value(self, params: Dict[str, Any]) -> float:
        """Get the raw evaluation value, either computed or from stored data."""
        if self.evaluate_from_data:
            return self._objective_function_from_data(params)
        else:
            return self.pure_objective_function(params)

    def _evaluate_with_timing(self, params: Dict[str, Any]) -> float:
        """Evaluate with sleep timing applied (overrides base class)."""
        time.sleep(self.sleep)
        raw_value = self._get_raw_value(params)
        return self.return_metric(raw_value)

    # Metrics that behave like scores (higher is better)
    SCORE_LIKE_METRICS = {"score", "accuracy", "r2", "f1", "precision", "recall", "auc"}

    def return_metric(self, score: float) -> float:
        """
        Transform raw score value based on metric setting.

        ML functions naturally return score values (higher is better).
        Supports standard metric names like 'accuracy', 'r2', 'f1', etc.
        """
        if self.metric in self.SCORE_LIKE_METRICS:
            return score
        elif self.metric == "loss":
            return -score
        else:
            # Treat unknown metrics as score-like (common for ML)
            return score

    def _to_loss(self, raw_value: float) -> float:
        """
        Convert raw value to loss (for minimization).

        ML functions naturally return score values (higher is better),
        so loss is the negated score.
        """
        return -raw_value

    def _to_score(self, raw_value: float) -> float:
        """
        Convert raw value to score (for maximization).

        ML functions naturally return score values,
        so this is an identity transformation.
        """
        return raw_value

    def evaluate(self, params):
        """
        Evaluate the function with given parameters.

        This method accepts both dict and list/tuple inputs, converting
        list/tuple to dict using param_names if available.

        Args:
            params: Either a dict of parameters or a list/tuple of values

        Returns:
            The objective function value

        Note:
            For ML functions, this overrides the positional-args evaluate()
            from BaseTestFunction to maintain backward compatibility with
            dict-based evaluation.
        """
        if isinstance(params, (list, tuple)):
            # Convert list/tuple to dict using param_names if available
            if hasattr(self, 'param_names'):
                param_dict = {name: val for name, val in zip(self.param_names, params)}
            else:
                # Fallback to generic naming
                param_dict = {f'x{i}': val for i, val in enumerate(params)}
        else:
            param_dict = params

        return self(param_dict)

    def _objective_function_from_data(self, params: Dict[str, Any]) -> float:
        """
        Evaluate function using stored search data for fast lookups.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Function evaluation result
            
        Raises:
            ValueError: If no stored data is found for the parameters
        """
        function_name = getattr(self, '_name_', self.__class__.__name__)
        
        # Look up stored evaluation
        result = self.search_data_manager.lookup_evaluation(function_name, params)
        if result is None:
            # No stored data found - could evaluate and store, or raise error
            raise ValueError(f"No stored evaluation found for parameters: {params}. "
                           f"Run data collection first for function: {function_name}")
        
        score, eval_time = result
        return score
    
    def _collect_search_data(self,
                           search_space: Optional[Dict[str, Any]] = None,
                           batch_size: int = 100,
                           verbose: bool = True) -> Dict[str, Any]:
        """
        Collect search data for this function across its parameter space.
        
        Args:
            search_space: Custom search space. If None, uses function's default
            batch_size: Number of evaluations to batch before database write
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with collection statistics
        """
        if not hasattr(self, 'data_collector'):
            self.search_data_manager = SearchDataManager()
            self.data_collector = SearchDataCollector(self.search_data_manager)
        
        return self.data_collector.collect_search_data(
            self, search_space, batch_size, verbose
        )
    
    def _get_search_data_status(self) -> Dict[str, Any]:
        """
        Get information about stored search data for this function.
        
        Returns:
            Dictionary with status information
        """
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
        """
        Get timing statistics for this function's evaluations.
        
        Returns:
            Dictionary with timing statistics
        """
        if not hasattr(self, 'data_collector'):
            self.search_data_manager = SearchDataManager()
            self.data_collector = SearchDataCollector(self.search_data_manager)
        
        return self.data_collector.get_timing_statistics(self)
