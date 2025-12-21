#!/usr/bin/env python3
"""
Automated plot generation script for all objective functions in Surfaces.
Generates both heatmap and surface plots for visualization in README.
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend

# Add src to path to import surfaces
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surfaces._visualize import matplotlib_heatmap, matplotlib_surface
from surfaces.test_functions.mathematical.test_functions_2d import *
from surfaces.test_functions.mathematical.test_functions_nd import *

# Ensure output directories exist
script_dir = os.path.dirname(os.path.abspath(__file__))
main_output_dir = os.path.join(script_dir, "..", "doc", "images")
mathematical_output_dir = os.path.join(main_output_dir, "mathematical")
os.makedirs(main_output_dir, exist_ok=True)
os.makedirs(mathematical_output_dir, exist_ok=True)

# Configuration for each function type
FUNCTION_CONFIGS = {
    # 2D Functions
    "AckleyFunction": {
        "search_space": {"x0": np.arange(-5, 5, 0.1), "x1": np.arange(-5, 5, 0.1)},
        "norm": None,
    },
    "BealeFunction": {
        "search_space": {"x0": np.arange(-4.5, 4.5, 0.1), "x1": np.arange(-4.5, 4.5, 0.1)},
        "norm": "color_log",
    },
    "BoothFunction": {
        "search_space": {"x0": np.arange(-10, 10, 0.2), "x1": np.arange(-10, 10, 0.2)},
        "norm": "color_log",
    },
    "BukinFunctionN6": {
        "search_space": {"x0": np.arange(-15, -5, 0.2), "x1": np.arange(-3, 3, 0.1)},
        "norm": "color_log",
    },
    "CrossInTrayFunction": {
        "search_space": {"x0": np.arange(-10, 10, 0.2), "x1": np.arange(-10, 10, 0.2)},
        "norm": None,
    },
    "DropWaveFunction": {
        "search_space": {"x0": np.arange(-5.2, 5.2, 0.1), "x1": np.arange(-5.2, 5.2, 0.1)},
        "norm": None,
    },
    "EasomFunction": {
        "search_space": {"x0": np.arange(-5, 5, 0.1), "x1": np.arange(-5, 5, 0.1)},
        "norm": None,
    },
    "EggholderFunction": {
        "search_space": {"x0": np.arange(-512, 512, 10), "x1": np.arange(-512, 512, 10)},
        "norm": None,
    },
    "GoldsteinPriceFunction": {
        "search_space": {"x0": np.arange(-2, 2, 0.05), "x1": np.arange(-2, 2, 0.05)},
        "norm": "color_log",
    },
    "HimmelblausFunction": {
        "search_space": {"x0": np.arange(-5, 5, 0.1), "x1": np.arange(-5, 5, 0.1)},
        "norm": "color_log",
    },
    "HölderTableFunction": {
        "search_space": {"x0": np.arange(-10, 10, 0.2), "x1": np.arange(-10, 10, 0.2)},
        "norm": None,
    },
    "LangermannFunction": {
        "search_space": {"x0": np.arange(0, 10, 0.1), "x1": np.arange(0, 10, 0.1)},
        "norm": None,
    },
    "LeviFunctionN13": {
        "search_space": {"x0": np.arange(-10, 10, 0.2), "x1": np.arange(-10, 10, 0.2)},
        "norm": "color_log",
    },
    "MatyasFunction": {
        "search_space": {"x0": np.arange(-10, 10, 0.2), "x1": np.arange(-10, 10, 0.2)},
        "norm": None,
    },
    "McCormickFunction": {
        "search_space": {"x0": np.arange(-1.5, 4, 0.1), "x1": np.arange(-3, 4, 0.1)},
        "norm": None,
    },
    "SchafferFunctionN2": {
        "search_space": {"x0": np.arange(-100, 100, 2), "x1": np.arange(-100, 100, 2)},
        "norm": None,
    },
    "SimionescuFunction": {
        "search_space": {"x0": np.arange(-1.25, 1.25, 0.05), "x1": np.arange(-1.25, 1.25, 0.05)},
        "norm": None,
    },
    "ThreeHumpCamelFunction": {
        "search_space": {"x0": np.arange(-5, 5, 0.1), "x1": np.arange(-5, 5, 0.1)},
        "norm": None,
    },
    # ND Functions (using 2D slices)
    "SphereFunction": {
        "search_space": {"x0": np.arange(-5, 5, 0.1), "x1": np.arange(-5, 5, 0.1)},
        "norm": None,
    },
    "RastriginFunction": {
        "search_space": {"x0": np.arange(-5, 5, 0.1), "x1": np.arange(-5, 5, 0.1)},
        "norm": None,
    },
    "RosenbrockFunction": {
        "search_space": {"x0": np.arange(-5, 5, 0.1), "x1": np.arange(-5, 5, 0.1)},
        "norm": "color_log",
    },
    "StyblinskiTangFunction": {
        "search_space": {"x0": np.arange(-5, 5, 0.1), "x1": np.arange(-5, 5, 0.1)},
        "norm": None,
    },
    "GriewankFunction": {
        "search_space": {"x0": np.arange(-10, 10, 0.2), "x1": np.arange(-10, 10, 0.2)},
        "norm": None,
    },
}


def get_function_instance(class_name):
    """Get an instance of the function class by name."""
    try:
        func_class = globals()[class_name]
        if class_name in [
            "SphereFunction",
            "RastriginFunction",
            "RosenbrockFunction",
            "StyblinskiTangFunction",
            "GriewankFunction",
        ]:
            # ND functions need n_dim parameter
            return func_class(n_dim=2, metric="loss")
        else:
            # 2D functions
            return func_class(metric="loss")
    except Exception as e:
        print(f"Error creating {class_name}: {e}")
        return None


def generate_plots():
    """Generate heatmap and surface plots for all configured functions."""
    print("Generating plots for mathematical objective functions...")

    for func_name, config in FUNCTION_CONFIGS.items():
        print(f"\nProcessing {func_name}...")

        try:
            # Get function instance
            func_instance = get_function_instance(func_name)
            if func_instance is None:
                continue

            search_space = config["search_space"]
            norm = config["norm"]

            # Generate file name
            file_name = (
                func_instance._name_ if hasattr(func_instance, "_name_") else func_name.lower()
            )

            # Generate heatmap
            print("  Generating heatmap...")
            heatmap_fig = matplotlib_heatmap(
                func_instance.objective_function,
                search_space,
                title=f"{func_instance.name} - Heatmap"
                if hasattr(func_instance, "name")
                else f"{func_name} - Heatmap",
                norm=norm,
            )
            heatmap_path = os.path.join(mathematical_output_dir, f"{file_name}_heatmap.jpg")
            heatmap_fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
            heatmap_fig.close()

            # Generate surface plot
            print("  Generating surface plot...")
            surface_fig = matplotlib_surface(
                func_instance.objective_function,
                search_space,
                title=f"{func_instance.name} - Surface"
                if hasattr(func_instance, "name")
                else f"{func_name} - Surface",
                norm=norm,
            )
            surface_path = os.path.join(mathematical_output_dir, f"{file_name}_surface.jpg")
            surface_fig.savefig(surface_path, dpi=150, bbox_inches="tight")
            surface_fig.close()

            print(f"  ✓ Generated plots for {func_name}")

        except Exception as e:
            print(f"  ✗ Error generating plots for {func_name}: {e}")
            continue

    print(
        f"\n✓ Mathematical function plot generation complete! Images saved to {mathematical_output_dir}"
    )

    # Generate ML function plots
    print("\nGenerating ML function plots...")
    try:
        import subprocess

        ml_script = os.path.join(script_dir, "generate_ml_plots.py")
        result = subprocess.run(
            [sys.executable, ml_script], capture_output=True, text=True, cwd=script_dir
        )

        if result.returncode == 0:
            print("✓ ML function plots generated successfully!")
            if result.stdout:
                print("ML Script Output:", result.stdout)
        else:
            print("✗ Error generating ML function plots")
            if result.stderr:
                print("Error:", result.stderr)
    except Exception as e:
        print(f"✗ Failed to run ML plot generation: {e}")


if __name__ == "__main__":
    generate_plots()
