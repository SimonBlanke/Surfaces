"""BBOB Sphere (f1) - Simplest separable function."""

from surfaces.test_functions.bbob import Sphere

# Create a 3-dimensional BBOB Sphere function
sphere = Sphere(n_dim=3, instance=1)

# Evaluate at a point
result = sphere({"x0": 0.0, "x1": 0.0, "x2": 0.0})
print(f"BBOB Sphere: {result:.4f}")
