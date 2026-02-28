import os

from surfaces._visualize import plot_fitness_distribution
from surfaces.test_functions.algebraic import GriewankFunction

func = GriewankFunction(n_dim=10)
fig = plot_fitness_distribution(func, n_samples=5000)

if not os.environ.get("SURFACES_TESTING"):
    fig.show()
