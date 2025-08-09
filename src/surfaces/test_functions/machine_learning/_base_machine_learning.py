# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
from typing import Dict, Any, Optional

from .._base_test_function import BaseTestFunction
from ...search_data_collection import SearchDataManager, SearchDataCollector


class MachineLearningFunction(BaseTestFunction):
    def __init__(self, metric="loss", sleep=0, evaluate_from_data=False, **kwargs):
        super().__init__(metric, sleep)
        
        self.evaluate_from_data = evaluate_from_data
        
        if evaluate_from_data:
            self.search_data_manager = SearchDataManager()
            self.data_collector = SearchDataCollector(self.search_data_manager)
            self._objective_function_ = self.objective_function_from_data
        else:
            self._objective_function_ = self.pure_objective_function
    
    def evaluate(self, params):
        """Evaluate the function with given parameters."""
        if isinstance(params, (list, tuple)):
            # Convert list/tuple to dict using param_names if available
            if hasattr(self, 'param_names'):
                param_dict = {name: val for name, val in zip(self.param_names, params)}
            else:
                # Fallback to generic naming
                param_dict = {f'x{i}': val for i, val in enumerate(params)}
        else:
            param_dict = params
        
        return self._objective_function_(param_dict)

    def objective_function_from_data(self, params: Dict[str, Any]) -> float:
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
    
    def collect_search_data(self, 
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
    
    def get_search_data_status(self) -> Dict[str, Any]:
        """
        Get information about stored search data for this function.
        
        Returns:
            Dictionary with status information
        """
        if not hasattr(self, 'data_collector'):
            self.search_data_manager = SearchDataManager()
            self.data_collector = SearchDataCollector(self.search_data_manager)
        
        return self.data_collector.get_collection_status(self)
    
    def clear_search_data(self) -> None:
        """Clear all stored search data for this function."""
        if not hasattr(self, 'search_data_manager'):
            self.search_data_manager = SearchDataManager()
        
        function_name = getattr(self, '_name_', self.__class__.__name__)
        self.search_data_manager.clear_data(function_name)
    
    def get_timing_statistics(self) -> Dict[str, float]:
        """
        Get timing statistics for this function's evaluations.
        
        Returns:
            Dictionary with timing statistics
        """
        if not hasattr(self, 'data_collector'):
            self.search_data_manager = SearchDataManager()
            self.data_collector = SearchDataCollector(self.search_data_manager)
        
        return self.data_collector.get_timing_statistics(self)
