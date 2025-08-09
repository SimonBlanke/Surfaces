#!/usr/bin/env python3
"""
Generate ML function visualizations showing hyperparameter interactions 
and dataset-specific performance patterns.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import plotly.io as pio
pio.kaleido.scope.mathjax = None  # Disable MathJax for faster rendering

# Add src to path to import surfaces
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from surfaces.test_functions.machine_learning.tabular.regression.test_functions import *
from surfaces.test_functions.machine_learning.tabular.classification.test_functions import *
from surfaces.visualize import (
    plotly_ml_hyperparameter_heatmap, 
    plotly_dataset_hyperparameter_analysis,
    create_ml_function_analysis_suite
)

# Ensure output directories exist
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, '..', 'doc', 'images')
ml_output_dir = os.path.join(output_dir, 'ml_functions')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ml_output_dir, exist_ok=True)

# ML Function configurations with reduced parameter spaces for faster evaluation
ML_FUNCTION_CONFIGS = {
    'GradientBoostingRegressorFunction': {
        'class': GradientBoostingRegressorFunction,
        'reduced_search_space': {
            'n_estimators': [10, 25, 50, 75, 100],  # Reduced from 3-150
            'max_depth': [2, 5, 10, 15],  # Reduced from 2-25
            'cv': [3, 5],  # Reduced from [2,3,4,5,8,10]
        },
        'hyperparameter_pairs': [
            ('n_estimators', 'max_depth'),
        ],
        'dataset_analyses': ['n_estimators', 'max_depth']
    },
    'KNeighborsRegressorFunction': {
        'class': KNeighborsRegressorFunction,
        'reduced_search_space': {
            'n_neighbors': [3, 8, 15, 25, 40],  # Reduced from 3-150
            'cv': [3, 5],
        },
        'hyperparameter_pairs': [],  # Only one main hyperparameter
        'dataset_analyses': ['n_neighbors']
    },
    'KNeighborsClassifierFunction': {
        'class': KNeighborsClassifierFunction,
        'reduced_search_space': {
            'n_neighbors': [3, 8, 15, 25, 40],  # Reduced from 3-150  
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],  # Reduced from 4 to 3
            'cv': [3, 5],
        },
        'hyperparameter_pairs': [
            ('n_neighbors', 'algorithm'),
        ],
        'dataset_analyses': ['n_neighbors', 'algorithm']
    }
}

def create_reduced_ml_function(func_class, reduced_space):
    """Create ML function instance with reduced search space for faster evaluation."""
    # Create instance with appropriate metric
    if 'Classifier' in func_class.__name__:
        func = func_class(metric="accuracy")  # Use accuracy for classification
    else:
        func = func_class(metric="neg_mean_squared_error")  # Use MSE for regression
    
    # Override search_space method with reduced space
    original_search_space = func.search_space
    def reduced_search_space_method(**kwargs):
        original_space = original_search_space(**kwargs)
        # Replace with reduced space, keeping datasets
        for param, values in reduced_space.items():
            original_space[param] = values
        return original_space
    
    func.search_space = reduced_search_space_method
    return func

def generate_ml_plots():
    """Generate all ML function visualization plots."""
    print("Generating ML function visualization plots...")
    
    for func_name, config in ML_FUNCTION_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"Processing {func_name}...")
        print(f"{'='*50}")
        
        # Create function instance with reduced search space
        func_class = config['class']
        reduced_space = config['reduced_search_space']
        ml_func = create_reduced_ml_function(func_class, reduced_space)
        
        # Create function-specific output directory
        func_output_dir = os.path.join(ml_output_dir, ml_func._name_)
        os.makedirs(func_output_dir, exist_ok=True)
        
        print(f"Search space: {ml_func.search_space()}")
        
        # 1. Generate hyperparameter vs hyperparameter plots
        print("\n1. Generating hyperparameter interaction plots...")
        for param1, param2 in config['hyperparameter_pairs']:
            print(f"  Creating {param1} vs {param2} plot...")
            
            fig = plotly_ml_hyperparameter_heatmap(
                ml_func, param1, param2,
                title=f"{ml_func.name} - {param1} vs {param2}"
            )
            
            # Save as image
            output_path = os.path.join(func_output_dir, f"{param1}_vs_{param2}_heatmap.jpg")
            fig.write_image(output_path, format="jpeg", width=900, height=700)
            
            # Also save to main images directory for README
            main_output_path = os.path.join(output_dir, f"{ml_func._name_}_{param1}_vs_{param2}_heatmap.jpg")
            fig.write_image(main_output_path, format="jpeg", width=900, height=700)
            
            print(f"    ✓ Saved {param1} vs {param2} heatmap")
        
        # 2. Generate dataset vs hyperparameter plots  
        print("\n2. Generating dataset analysis plots...")
        for hyperparameter in config['dataset_analyses']:
            print(f"  Creating dataset vs {hyperparameter} plot...")
            
            fig = plotly_dataset_hyperparameter_analysis(
                ml_func, hyperparameter,
                title=f"{ml_func.name} - Dataset vs {hyperparameter}"
            )
            
            # Save as image
            output_path = os.path.join(func_output_dir, f"dataset_vs_{hyperparameter}_analysis.jpg")
            fig.write_image(output_path, format="jpeg", width=1000, height=700)
            
            # Also save to main images directory for README
            main_output_path = os.path.join(output_dir, f"{ml_func._name_}_dataset_vs_{hyperparameter}_analysis.jpg")
            fig.write_image(main_output_path, format="jpeg", width=1000, height=700)
            
            print(f"    ✓ Saved dataset vs {hyperparameter} analysis")
                
        print(f"✓ Completed {func_name} analysis")
    
    print(f"\n✓ ML plot generation complete! Images saved to:")
    print(f"  - Individual function plots: {ml_output_dir}")
    print(f"  - README plots: {output_dir}")

def generate_sample_plots():
    """Generate a few sample plots for testing."""
    print("Generating sample ML plots for testing...")
    
    # Create with proper metric
    ml_func = KNeighborsClassifierFunction(metric="accuracy")
    
    # Override search space with small values for testing
    def reduced_search_space_method(**kwargs):
        return {
            'n_neighbors': [3, 10, 20],  # Very small for testing
            'algorithm': ['auto', 'ball_tree'], 
            'cv': [3],
            'dataset': ml_func.dataset_default  # Keep all datasets
        }
    
    ml_func.search_space = reduced_search_space_method
    
    print("Creating sample hyperparameter plot...")
    fig1 = plotly_ml_hyperparameter_heatmap(
        ml_func, 'n_neighbors', 'algorithm',
        title="Sample: KNN Hyperparameter Analysis"
    )
    fig1.write_image(os.path.join(output_dir, "sample_knn_hyperparams.jpg"), 
                    format="jpeg", width=900, height=700)
    
    print("Creating sample dataset analysis plot...")
    fig2 = plotly_dataset_hyperparameter_analysis(
        ml_func, 'n_neighbors',
        title="Sample: Dataset vs n_neighbors"
    )
    fig2.write_image(os.path.join(output_dir, "sample_knn_datasets.jpg"), 
                    format="jpeg", width=1000, height=700)
    
    print("✓ Sample plots generated successfully!")

if __name__ == "__main__":
    # Check if we should generate sample plots first (faster for testing)
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        generate_sample_plots()
    else:
        generate_ml_plots()