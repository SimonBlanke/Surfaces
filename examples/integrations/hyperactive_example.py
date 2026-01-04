"""Hyperactive Integration - Optimize a Surfaces test function."""

import numpy as np
from hyperactive import Hyperactive

from surfaces.test_functions import RastriginFunction

# Create a 3-dimensional Rastrigin function
rastrigin = RastriginFunction(n_dim=3)

# Define the search space
search_space = {
    "x0": np.linspace(-5, 5, 100),
    "x1": np.linspace(-5, 5, 100),
    "x2": np.linspace(-5, 5, 100),
}


# Objective function (Hyperactive maximizes by default)
def objective(params):
    return -rastrigin(params)  # Negate for minimization


# Run optimization
hyper = Hyperactive()
hyper.add_search(objective, search_space, n_iter=100)
hyper.run()

# Results
print(f"Best score: {-hyper.best_score(objective):.6f}")
print(f"Best params: {hyper.best_para(objective)}")
