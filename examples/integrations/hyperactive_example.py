"""Hyperactive Integration - Optimize a Surfaces test function."""

import numpy as np

from hyperactive.experiment.func import FunctionExperiment
from hyperactive.opt.gfo import HillClimbing

from surfaces.test_functions import RastriginFunction

# Create a 3-dimensional Rastrigin function
rastrigin = RastriginFunction(n_dim=3)


# Objective function (Hyperactive maximizes, so negate for minimization)
def objective(params):
    return -rastrigin(params)


# Define the search space
search_space = {
    "x0": np.linspace(-5, 5, 101).tolist(),
    "x1": np.linspace(-5, 5, 101).tolist(),
    "x2": np.linspace(-5, 5, 101).tolist(),
}

# Create experiment and optimizer
experiment = FunctionExperiment(objective)
optimizer = HillClimbing(
    search_space=search_space,
    n_iter=100,
    experiment=experiment,
)

# Run optimization
best_params = optimizer.solve()
best_value = rastrigin(best_params)

print(f"Best value: {best_value:.6f}")
print(f"Best params: {best_params}")
