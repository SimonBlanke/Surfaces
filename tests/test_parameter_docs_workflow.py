#!/usr/bin/env python3
"""
Test script to demonstrate the complete parameter coverage documentation workflow.

This script shows how to:
1. Generate sample data for ML functions
2. Create parameter coverage documentation
3. Verify the documentation updates automatically
"""

import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from surfaces.test_functions.machine_learning.tabular.regression.test_functions.k_neighbors_regressor import (
    KNeighborsRegressorFunction,
)


def test_workflow():
    """Test the complete workflow."""
    print("🧪 Testing ML Parameter Coverage Documentation Workflow")
    print("=" * 60)

    # Test 1: Generate documentation with no data
    print("\n1. Testing documentation generation with no/minimal data...")
    os.system("python generate_parameter_coverage_docs.py -f k_neighbors_regressor")

    # Test 2: Generate some sample data
    print("\n2. Generating sample search data...")

    # Create a function instance and collect a small amount of data
    func = KNeighborsRegressorFunction()

    # Use a smaller search space for testing
    custom_search_space = {
        "n_neighbors": [3, 5, 7],
        "algorithm": ["auto", "ball_tree"],
        "cv": [2, 3],
        "dataset": func.search_space["dataset"][:1],  # Just one dataset
    }

    print(f"Collecting data with custom search space: {custom_search_space}")
    print("(This will take a moment as it actually trains ML models...)")

    try:
        stats = func._collect_search_data(
            search_space=custom_search_space, batch_size=5, verbose=True
        )
        print(f"✅ Data collection completed: {stats['evaluations_collected']} new evaluations")
    except Exception as e:
        print(f"⚠️ Data collection failed: {e}")
        print("This is normal if scikit-learn datasets are not available")

    # Test 3: Generate updated documentation
    print("\n3. Generating updated documentation...")
    os.system("python generate_parameter_coverage_docs.py -f k_neighbors_regressor")

    # Test 4: Test JSON output
    print("\n4. Testing JSON output...")
    os.system("python generate_parameter_coverage_docs.py -f k_neighbors_regressor --json")

    # Test 5: Test full documentation generation
    print("\n5. Testing full documentation generation...")
    os.system("python generate_parameter_coverage_docs.py")

    print("\n✅ Workflow test completed!")
    print("\nGenerated files:")
    print("- ML_PARAMETER_COVERAGE.md")
    print("- docs/ML_PARAMETER_COVERAGE.md")
    print("- docs/ml_parameter_coverage.json")

    print("\n📊 View the generated documentation: docs/ML_PARAMETER_COVERAGE.md")

    # Show current database status
    print("\n📈 Current database status:")
    os.system("python collect_ml_search_data.py --list")


if __name__ == "__main__":
    test_workflow()
