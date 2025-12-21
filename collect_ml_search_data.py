#!/usr/bin/env python3
"""
Command-line utility for collecting search data for machine learning test functions.

This script allows users to easily collect search data for all ML functions
or specific ones, with options for batch size and verbosity.
"""

import argparse
import os
import sys
import time

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import available ML test functions
from surfaces.test_functions.machine_learning.tabular.classification.test_functions.k_neighbors_classifier import (
    KNeighborsClassifierFunction,
)
from surfaces.test_functions.machine_learning.tabular.regression.test_functions.gradient_boosting_regressor import (
    GradientBoostingRegressorFunction,
)
from surfaces.test_functions.machine_learning.tabular.regression.test_functions.k_neighbors_regressor import (
    KNeighborsRegressorFunction,
)

# Available ML functions
ML_FUNCTIONS = {
    "gradient_boosting_regressor": GradientBoostingRegressorFunction,
    "k_neighbors_regressor": KNeighborsRegressorFunction,
    "k_neighbors_classifier": KNeighborsClassifierFunction,
}


def collect_data_for_function(func_class, func_name, batch_size, verbose, custom_search_space=None):
    """Collect search data for a single function."""
    print(f"\n{'=' * 60}")
    print(f"Collecting data for: {func_name}")
    print(f"{'=' * 60}")

    # Create function instance
    func = func_class()

    # Show current status
    status = func.get_search_data_status()
    print(
        f"Current coverage: {status['completion_percentage']:.2f}% "
        f"({status['stored_evaluations']}/{status['total_combinations']})"
    )

    if status["completion_percentage"] >= 100:
        print("Data collection already complete!")
        return status

    # Collect data
    start_time = time.time()

    try:
        stats = func.collect_search_data(
            search_space=custom_search_space, batch_size=batch_size, verbose=verbose
        )

        collection_time = time.time() - start_time

        print("\nCollection Summary:")
        print(f"- New evaluations: {stats['evaluations_collected']}")
        print(f"- Total stored: {stats['total_evaluations_stored']}")
        print(f"- Collection time: {collection_time:.2f} seconds")

        if stats["evaluations_collected"] > 0:
            avg_time = stats["collection_time_seconds"] / stats["evaluations_collected"]
            print(f"- Average evaluation time: {avg_time:.4f} seconds")

        # Show timing statistics
        timing_stats = func.get_timing_statistics()
        print(
            f"- Min/Max evaluation time: {timing_stats['min_time']:.4f}s / {timing_stats['max_time']:.4f}s"
        )

        return stats

    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
        return None
    except Exception as e:
        print(f"Error during collection: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Collect search data for machine learning test functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available functions:
{chr(10).join(f"  - {name}" for name in ML_FUNCTIONS.keys())}

Examples:
  python collect_ml_search_data.py --all                    # Collect for all functions
  python collect_ml_search_data.py gradient_boosting_regressor  # Collect for specific function
  python collect_ml_search_data.py --list                   # Show collection status
  python collect_ml_search_data.py --batch-size 50 --all    # Use custom batch size
""",
    )

    parser.add_argument(
        "functions",
        nargs="*",
        choices=list(ML_FUNCTIONS.keys()),
        help="Specific functions to collect data for",
    )

    parser.add_argument(
        "--all", action="store_true", help="Collect data for all available functions"
    )

    parser.add_argument(
        "--list", action="store_true", help="Show collection status for all functions"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of evaluations to batch before database write (default: 100)",
    )

    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    parser.add_argument(
        "--clear", metavar="FUNCTION", help="Clear stored data for a specific function"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Handle clear command
    if args.clear:
        if args.clear not in ML_FUNCTIONS:
            print(f"Error: Unknown function '{args.clear}'")
            return 1

        func = ML_FUNCTIONS[args.clear]()
        func.clear_search_data()
        print(f"Cleared search data for: {args.clear}")
        return 0

    # Handle list command
    if args.list:
        print("Search Data Collection Status")
        print("=" * 50)

        for name, func_class in ML_FUNCTIONS.items():
            func = func_class()
            status = func.get_search_data_status()
            db_info = status["database_info"]

            print(f"\n{name}:")
            print(
                f"  Coverage: {status['completion_percentage']:.2f}% "
                f"({status['stored_evaluations']}/{status['total_combinations']})"
            )

            if db_info.get("exists", False):
                size_mb = db_info.get("size_bytes", 0) / (1024 * 1024)
                print(f"  Database size: {size_mb:.2f} MB")

                # Show timing stats if available
                timing = func.get_timing_statistics()
                if timing["count"] > 0:
                    print(f"  Avg evaluation time: {timing['average_time']:.4f}s")
            else:
                print("  No data collected yet")

        return 0

    # Determine which functions to collect for
    target_functions = []

    if args.all:
        target_functions = list(ML_FUNCTIONS.items())
    elif args.functions:
        target_functions = [(name, ML_FUNCTIONS[name]) for name in args.functions]
    else:
        parser.print_help()
        return 1

    # Collect data
    print(f"Starting search data collection for {len(target_functions)} function(s)")

    total_start_time = time.time()
    results = {}

    for func_name, func_class in target_functions:
        result = collect_data_for_function(func_class, func_name, args.batch_size, verbose)
        results[func_name] = result

    total_time = time.time() - total_start_time

    # Summary
    print(f"\n{'=' * 60}")
    print("COLLECTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total time: {total_time:.2f} seconds")

    total_collected = sum(r["evaluations_collected"] for r in results.values() if r)
    print(f"Total new evaluations: {total_collected}")

    for func_name, result in results.items():
        if result:
            print(f"- {func_name}: {result['evaluations_collected']} evaluations")
        else:
            print(f"- {func_name}: Failed or interrupted")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
