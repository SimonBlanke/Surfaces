# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from .search_data_manager import SearchDataManager
from .grid_generator import GridGenerator


class SearchDataCollector:
    """
    Collects search data by evaluating machine learning functions across parameter grids.
    
    This class coordinates the evaluation of ML functions across all parameter combinations,
    measures execution time for benchmarking, and stores results for future fast lookups.
    """
    
    def __init__(self, data_manager: Optional[SearchDataManager] = None):
        """
        Initialize the SearchDataCollector.
        
        Args:
            data_manager: SearchDataManager instance. If None, creates a default one.
        """
        self.data_manager = data_manager or SearchDataManager()
        self.grid_generator = GridGenerator()
    
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
        
        if search_space is None:
            search_space = test_function.search_space()
        
        # Validate search space
        validation_errors = self.grid_generator.validate_search_space(search_space)
        if validation_errors:
            raise ValueError(f"Invalid search space: {'; '.join(validation_errors)}")
        
        # Get parameter names and total combinations
        parameter_names = list(search_space.keys())
        total_combinations = self.grid_generator.count_combinations(search_space)
        
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
        
        # Collect evaluations
        start_time = time.time()
        collected_count = 0
        batch_evaluations = []
        
        for i, parameters in enumerate(self.grid_generator.generate_grid_iterator(search_space)):
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