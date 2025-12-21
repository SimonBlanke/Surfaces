#!/usr/bin/env python3
"""
Example of ML function visualization capabilities.
Shows how to create hyperparameter analysis plots and dataset comparisons.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surfaces._visualize import (
    plotly_dataset_hyperparameter_analysis,
    plotly_ml_hyperparameter_heatmap,
)
from surfaces.test_functions.machine_learning.tabular.classification.test_functions import (
    KNeighborsClassifierFunction,
)


def main():
    print("ML Function Visualization Example")
    print("=" * 40)

    # Create KNN classifier function
    knn_func = KNeighborsClassifierFunction(metric="accuracy")

    # Override search space with smaller values for faster demo
    def reduced_search_space(**kwargs):
        return {
            "n_neighbors": [3, 5, 10, 15, 20],
            "algorithm": ["auto", "ball_tree", "kd_tree"],
            "cv": [3, 5],
            "dataset": knn_func.search_space()["dataset"],  # Keep all datasets
        }

    knn_func.search_space = reduced_search_space

    print("Search space:", knn_func.search_space())

    # Example 1: Hyperparameter interaction analysis
    print("\n1. Creating hyperparameter interaction plot...")
    print("   This shows how n_neighbors and algorithm interact")
    try:
        fig1 = plotly_ml_hyperparameter_heatmap(
            knn_func, "n_neighbors", "algorithm", title="KNN: n_neighbors vs algorithm"
        )

        # Save plot
        output_path = os.path.join(os.path.dirname(__file__), "knn_hyperparams_demo.html")
        fig1.write_html(output_path)
        print(f"   ✓ Saved to: {output_path}")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 2: Dataset sensitivity analysis
    print("\n2. Creating dataset vs hyperparameter analysis...")
    print("   This shows how n_neighbors affects performance across datasets")
    try:
        fig2 = plotly_dataset_hyperparameter_analysis(
            knn_func, "n_neighbors", title="KNN: Dataset Performance vs n_neighbors"
        )

        # Save plot
        output_path = os.path.join(os.path.dirname(__file__), "knn_datasets_demo.html")
        fig2.write_html(output_path)
        print(f"   ✓ Saved to: {output_path}")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 40)
    print("Demo complete! Check the generated HTML files.")
    print("\nKey insights from ML visualizations:")
    print("• Hyperparameter interactions show optimal combinations")
    print("• Dataset analysis reveals which parameters are most sensitive")
    print("• Different datasets may require different hyperparameter values")
    print("• Some algorithms perform consistently across datasets")


if __name__ == "__main__":
    main()
