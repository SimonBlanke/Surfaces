#!/usr/bin/env python3
"""
Example demonstrating the new search data collection feature.

This example shows how to:
1. Collect search data for a machine learning test function
2. Use the stored data for fast evaluations
3. Get timing statistics for benchmarking
"""

import sys
import os
import time

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from surfaces.test_functions.machine_learning.tabular.regression.test_functions.gradient_boosting_regressor import GradientBoostingRegressorFunction


def main():
    print("Search Data Collection Example")
    print("=" * 40)
    
    # Create a machine learning test function
    ml_func = GradientBoostingRegressorFunction()
    
    # Define a small search space for demo purposes
    # (In practice, you'd use the full default search space)
    demo_search_space = {
        'n_estimators': [10, 20, 50],
        'max_depth': [3, 5, 7],
        'cv': [2, 3],
        'dataset': ml_func.dataset_default[:1]  # Use first dataset only
    }
    
    print(f"Demo search space has {3 * 3 * 2 * 1} = 18 combinations")
    print()
    
    # 1. Collect search data
    print("Step 1: Collecting search data...")
    print("-" * 30)
    start_time = time.time()
    
    collection_stats = ml_func.collect_search_data(
        search_space=demo_search_space,
        batch_size=10,
        verbose=True
    )
    
    collection_time = time.time() - start_time
    print(f"Total collection time: {collection_time:.2f} seconds")
    print()
    
    # 2. Show timing statistics
    print("Step 2: Timing statistics for benchmarking")
    print("-" * 30)
    timing_stats = ml_func.get_timing_statistics()
    print(f"Average evaluation time: {timing_stats['average_time']:.4f} seconds")
    print(f"Min evaluation time: {timing_stats['min_time']:.4f} seconds")
    print(f"Max evaluation time: {timing_stats['max_time']:.4f} seconds")
    print(f"Total evaluation time: {timing_stats['total_time']:.4f} seconds")
    print()
    
    # 3. Use stored data for fast evaluations
    print("Step 3: Fast evaluation using stored data")
    print("-" * 30)
    
    # Create a new instance that uses stored data
    fast_func = GradientBoostingRegressorFunction(evaluate_from_data=True)
    
    # Test parameters that should exist in our stored data
    test_params = {
        'n_estimators': 20,
        'max_depth': 5,
        'cv': 3,
        'dataset': ml_func.dataset_default[0]
    }
    
    # Time the fast lookup
    lookup_start = time.time()
    result = fast_func.objective_function_from_data(test_params)
    lookup_time = time.time() - lookup_start
    
    print(f"Parameters: {test_params}")
    print(f"Result: {result:.6f}")
    print(f"Lookup time: {lookup_time:.6f} seconds")
    print(f"Speedup factor: ~{timing_stats['average_time']/max(lookup_time, 0.000001):.0f}x")
    print()
    
    # 4. Show collection status
    print("Step 4: Collection status")
    print("-" * 30)
    status = ml_func.get_search_data_status()
    print(f"Function: {status['function_name']}")
    print(f"Stored evaluations: {status['stored_evaluations']}")
    print(f"Total possible combinations: {status['total_combinations']}")
    print(f"Coverage: {status['completion_percentage']:.2f}%")
    print()
    
    # 5. Demonstrate benchmarking scenario
    print("Step 5: Benchmarking scenario")
    print("-" * 30)
    print("Imagine you want to test optimization algorithms within a time budget...")
    
    # Simulate evaluating multiple parameter combinations
    test_combinations = [
        {'n_estimators': 10, 'max_depth': 3, 'cv': 2, 'dataset': ml_func.dataset_default[0]},
        {'n_estimators': 20, 'max_depth': 5, 'cv': 2, 'dataset': ml_func.dataset_default[0]},
        {'n_estimators': 50, 'max_depth': 7, 'cv': 3, 'dataset': ml_func.dataset_default[0]},
    ]
    
    total_benchmark_time = 0
    for params in test_combinations:
        lookup_start = time.time()
        result = fast_func.objective_function_from_data(params)
        lookup_time = time.time() - lookup_start
        total_benchmark_time += lookup_time
        print(f"Evaluated {params['n_estimators']}/{params['max_depth']}: {result:.4f} ({lookup_time:.6f}s)")
    
    print(f"Total benchmark time: {total_benchmark_time:.6f} seconds")
    print(f"Would have taken ~{timing_stats['average_time'] * len(test_combinations):.4f}s without stored data")
    print()
    
    # Clean up (optional - comment out to keep data for future use)
    print("Cleaning up stored data...")
    ml_func.clear_search_data()
    print("Done!")


if __name__ == "__main__":
    main()