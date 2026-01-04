import os

from surfaces.test_functions.algebraic.test_functions_nd import GriewankFunction
from surfaces.visualize import plot_fitness_distribution

func = GriewankFunction(n_dim=10)
fig = plot_fitness_distribution(func, n_samples=5000)

if not os.environ.get("SURFACES_TESTING"):
    fig.show()
