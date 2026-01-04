"""Sphere Function - The simplest convex benchmark function."""

from surfaces.test_functions import SphereFunction

# Create a 3-dimensional Sphere function
sphere = SphereFunction(n_dim=3)

# Evaluate at a point: f(x) = sum(x_i^2)
result = sphere({"x0": 1.0, "x1": 2.0, "x2": 3.0})
print(f"Sphere(1, 2, 3) = {result}")  # Output: 14.0
