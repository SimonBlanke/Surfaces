"""Sphere Function - The simplest convex benchmark function."""

from surfaces.test_functions import SphereFunction

sphere = SphereFunction(n_dim=3)

# First evaluation
params1 = {"x0": 1.0, "x1": 2.0, "x2": 3.0}
score1 = sphere(params1)
print(f"params: {params1} -> score: {score1}")

# Second evaluation
params2 = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
score2 = sphere(params2)
print(f"params: {params2} -> score: {score2}")
