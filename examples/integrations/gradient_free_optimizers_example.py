"""Gradient-Free-Optimizers Integration - Optimize a Surfaces test function."""

import numpy as np
from gradient_free_optimizers import HillClimbingOptimizer

from surfaces.test_functions.algebraic import SphereFunction

# Create a 5-dimensional Sphere function
sphere = SphereFunction(n_dim=5)

# Define the search space for GFO (numpy arrays)
search_space = {
    "x0": np.linspace(-5, 5, 100),
    "x1": np.linspace(-5, 5, 100),
    "x2": np.linspace(-5, 5, 100),
    "x3": np.linspace(-5, 5, 100),
    "x4": np.linspace(-5, 5, 100),
}


# Objective function (GFO minimizes by default)
def objective(params):
    return -sphere(params)  # Negate because GFO maximizes


# Run optimization
opt = HillClimbingOptimizer(search_space)
opt.search(objective, n_iter=100)

# Results
print(f"Best score: {-opt.best_score:.6f}")
print(f"Best params: {opt.best_para}")
