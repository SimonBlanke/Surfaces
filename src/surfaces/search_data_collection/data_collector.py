# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from .search_data_manager import SearchDataManager
from .grid_generator import GridGenerator


class DatasetRegistry:
    """
    Registry for dataset functions to handle object serialization in search spaces.
    
    Maps string identifiers to dataset functions, allowing datasets to be stored
    in the database as strings and resolved back to callable functions.
    """
    
    def __init__(self):
        self._datasets = {}
        self._register_default_datasets()
    
    def _register_default_datasets(self):
        """Register default datasets from the surfaces package."""
        # Import classification datasets
        from ..test_functions.machine_learning.tabular.classification.datasets import (
            digits_data, wine_data, iris_data
        )
        self.register("digits_data", digits_data)
        self.register("wine_data", wine_data) 
        self.register("iris_data", iris_data)
        
        # Import regression datasets
        from ..test_functions.machine_learning.tabular.regression.datasets import (
            diabetes_data
        )
        self.register("diabetes_data", diabetes_data)
    
    def register(self, name: str, dataset_func: Callable):
        """Register a dataset function with a string identifier."""
        self._datasets[name] = dataset_func
    
    def get(self, name: str) -> Callable:
        """Get a dataset function by its string identifier."""
        if name not in self._datasets:
            raise ValueError(f"Dataset '{name}' not found in registry. Available: {list(self._datasets.keys())}")
        return self._datasets[name]
    
    def get_name(self, dataset_func: Callable) -> str:
        """Get the string identifier for a dataset function."""
        for name, func in self._datasets.items():
            if func is dataset_func:
                return name
        # If not found, use the function name as fallback
        return getattr(dataset_func, '__name__', str(dataset_func))
    
    def list_datasets(self) -> List[str]:
        """List all registered dataset names."""
        return list(self._datasets.keys())


class SearchDataCollector:
    """
    Collects search data by evaluating machine learning functions across parameter grids.
    
    Enhanced features:
    - Automatic use of default search spaces from ML functions
    - Intelligent handling of dataset objects through string serialization
    - Incremental database updates
    """
    
    def __init__(self, data_manager: Optional[SearchDataManager] = None):
        """
        Initialize the SearchDataCollector.
        
        Args:
            data_manager: SearchDataManager instance. If None, creates a default one.
        """
        self.data_manager = data_manager or SearchDataManager()
        self.grid_generator = GridGenerator()
        self.dataset_registry = DatasetRegistry()
    
    def collect_search_data(self, 
                          test_function, 
                          search_space: Optional[Dict[str, List[Any]]] = None,
                          batch_size: int = 100,
                          verbose: bool = True) -> Dict[str, Any]:
        """
        Collect search data for a test function across its parameter grid.
        
        Args:
            test_function: Machine learning test function instance
            search_space: Custom search space. If None, uses function's default
            batch_size: Number of evaluations to batch before database write
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with collection statistics
        """
        # Get function name and search space
        function_name = getattr(test_function, '_name_', test_function.__class__.__name__)
        
        # Use default search space if none provided
        if search_space is None:
            search_space = test_function.search_space()
        
        # Process search space to handle dataset objects
        processed_search_space = self._process_search_space(search_space)
        
        # Validate search space
        validation_errors = self.grid_generator.validate_search_space(processed_search_space)
        if validation_errors:
            raise ValueError(f"Invalid search space: {'; '.join(validation_errors)}")
        
        # Get parameter names and total combinations
        parameter_names = list(processed_search_space.keys())
        total_combinations = self.grid_generator.count_combinations(processed_search_space)
        
        if verbose:
            print(f"Collecting search data for: {function_name}")
            print(f"Parameters: {parameter_names}")
            print(f"Total combinations: {total_combinations}")
        
        # Create database table
        self.data_manager.create_table(function_name, parameter_names)
        
        # Check for existing data
        existing_count = len(self.data_manager.get_all_evaluations(function_name))
        if existing_count > 0:
            if verbose:
                print(f"Found {existing_count} existing evaluations")
        
        # Create wrapper function that handles dataset resolution
        original_objective = test_function.pure_objective_function
        wrapper_objective = self._create_dataset_wrapper(original_objective)
        test_function.pure_objective_function = wrapper_objective
        
        # Collect evaluations
        start_time = time.time()
        collected_count = 0
        batch_evaluations = []
        
        try:
            for i, parameters in enumerate(self.grid_generator.generate_grid_iterator(processed_search_space)):
                # Check if evaluation already exists
                existing = self.data_manager.lookup_evaluation(function_name, parameters)
                if existing is not None:
                    continue
                
                # Evaluate function with timing
                eval_start = time.time()
                try:
                    score = test_function.pure_objective_function(parameters)
                except Exception as e:
                    if verbose:
                        print(f"Error evaluating parameters {parameters}: {e}")
                    continue
                eval_time = time.time() - eval_start
                
                # Add to batch
                batch_evaluations.append((parameters, score, eval_time))
                collected_count += 1
                
                # Store batch when it reaches batch_size
                if len(batch_evaluations) >= batch_size:
                    self.data_manager.store_batch(function_name, parameter_names, batch_evaluations)
                    batch_evaluations = []
                    
                    if verbose:
                        progress = (i + 1) / total_combinations * 100
                        print(f"Progress: {progress:.1f}% ({i + 1}/{total_combinations})")
            
            # Store remaining evaluations
            if batch_evaluations:
                self.data_manager.store_batch(function_name, parameter_names, batch_evaluations)
        
        except Exception as e:
            # Re-raise the exception after cleanup
            raise
        finally:
            # Always restore original objective function
            test_function.pure_objective_function = original_objective
        
        total_time = time.time() - start_time
        
        # Final statistics
        final_count = len(self.data_manager.get_all_evaluations(function_name))
        
        stats = {
            "function_name": function_name,
            "total_combinations": total_combinations,
            "evaluations_collected": collected_count,
            "total_evaluations_stored": final_count,
            "collection_time_seconds": total_time,
            "average_evaluation_time": total_time / max(collected_count, 1)
        }
        
        if verbose:
            print(f"Collection completed!")
            print(f"New evaluations: {collected_count}")
            print(f"Total stored: {final_count}")
            print(f"Collection time: {total_time:.2f} seconds")
            if collected_count > 0:
                print(f"Average evaluation time: {total_time/collected_count:.4f} seconds")
        
        return stats
    
    def evaluate_with_lookup(self, 
                           test_function,
                           parameters: Dict[str, Any]) -> Tuple[float, float]:
        """
        Evaluate parameters using stored data if available, otherwise compute.
        
        Args:
            test_function: Machine learning test function instance
            parameters: Parameters to evaluate
            
        Returns:
            Tuple of (score, evaluation_time)
        """
        function_name = getattr(test_function, '_name_', test_function.__class__.__name__)
        
        # Try to lookup existing evaluation
        result = self.data_manager.lookup_evaluation(function_name, parameters)
        if result is not None:
            return result
        
        # Evaluate and store if not found
        eval_start = time.time()
        score = test_function.pure_objective_function(parameters)
        eval_time = time.time() - eval_start
        
        # Store for future use
        self.data_manager.store_evaluation(function_name, parameters, score, eval_time)
        
        return score, eval_time
    
    def get_collection_status(self, test_function) -> Dict[str, Any]:
        """
        Get status information about data collection for a function.
        
        Args:
            test_function: Machine learning test function instance
            
        Returns:
            Dictionary with status information
        """
        function_name = getattr(test_function, '_name_', test_function.__class__.__name__)
        search_space = test_function.search_space()
        
        total_combinations = self.grid_generator.count_combinations(search_space)
        stored_evaluations = len(self.data_manager.get_all_evaluations(function_name))
        
        completion_percentage = (stored_evaluations / max(total_combinations, 1)) * 100
        
        return {
            "function_name": function_name,
            "total_combinations": total_combinations,
            "stored_evaluations": stored_evaluations,
            "completion_percentage": completion_percentage,
            "database_info": self.data_manager.get_database_info(function_name)
        }
    
    def clear_function_data(self, test_function) -> None:
        """
        Clear all stored data for a function.
        
        Args:
            test_function: Machine learning test function instance
        """
        function_name = getattr(test_function, '_name_', test_function.__class__.__name__)
        self.data_manager.clear_data(function_name)
    
    def get_timing_statistics(self, test_function) -> Dict[str, float]:
        """
        Get timing statistics for a function's evaluations.
        
        Args:
            test_function: Machine learning test function instance
            
        Returns:
            Dictionary with timing statistics
        """
        function_name = getattr(test_function, '_name_', test_function.__class__.__name__)
        evaluations = self.data_manager.get_all_evaluations(function_name)
        
        if not evaluations:
            return {"count": 0}
        
        eval_times = [eval_data["evaluation_time"] for eval_data in evaluations]
        
        return {
            "count": len(eval_times),
            "total_time": sum(eval_times),
            "average_time": sum(eval_times) / len(eval_times),
            "min_time": min(eval_times),
            "max_time": max(eval_times)
        }
    
    def _process_search_space(self, search_space: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Process search space to handle dataset objects.
        
        Converts callable dataset objects to string identifiers that can be stored in database.
        """
        processed = {}
        
        for param_name, param_values in search_space.items():
            if param_name == "dataset":
                # Convert dataset functions to string identifiers
                processed_values = []
                for value in param_values:
                    if callable(value):
                        dataset_name = self.dataset_registry.get_name(value)
                        processed_values.append(dataset_name)
                    else:
                        processed_values.append(value)
                processed[param_name] = processed_values
            else:
                processed[param_name] = param_values
        
        return processed
    
    def _create_dataset_wrapper(self, original_objective: Callable) -> Callable:
        """
        Create a wrapper function that resolves dataset string identifiers back to functions.
        """
        def wrapper(params):
            processed_params = params.copy()
            
            # Resolve dataset string back to function if present
            if "dataset" in processed_params:
                dataset_value = processed_params["dataset"]
                if isinstance(dataset_value, str):
                    processed_params["dataset"] = self.dataset_registry.get(dataset_value)
            
            return original_objective(processed_params)
        
        return wrapper
    
    def register_dataset(self, name: str, dataset_func: Callable):
        """Register a custom dataset function."""
        self.dataset_registry.register(name, dataset_func)