"""BBOB Sphere (f1) - Simplest separable function."""

from surfaces.test_functions.benchmark.bbob import Sphere

sphere = Sphere(n_dim=3, instance=1)

# First evaluation
params1 = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
score1 = sphere(params1)
print(f"params: {params1} -> score: {score1:.4f}")

# Second evaluation
params2 = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
score2 = sphere(params2)
print(f"params: {params2} -> score: {score2:.4f}")
