"""SciPy Integration - Optimize a Surfaces test function."""

import numpy as np
from scipy.optimize import minimize

from surfaces.test_functions import RosenbrockFunction

# Create a 3-dimensional Rosenbrock function
rosenbrock = RosenbrockFunction(n_dim=3)


def objective(x):
    """Wrapper for scipy (expects array input)."""
    params = {f"x{i}": x[i] for i in range(len(x))}
    return rosenbrock(params)


# Initial guess
x0 = np.array([0.0, 0.0, 0.0])

# Bounds for each dimension
bounds = [(-5, 5), (-5, 5), (-5, 5)]

# Run optimization
result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

# Results
print(f"Best value: {result.fun:.6f}")
print(f"Best params: {result.x}")
print(f"Converged: {result.success}")
