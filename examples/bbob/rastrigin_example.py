"""BBOB Rastrigin Rotated (f15) - Multimodal with adequate global structure."""

from surfaces.test_functions.bbob import RastriginRotated

# Create a 2-dimensional rotated Rastrigin function
rastrigin = RastriginRotated(n_dim=2, instance=1)

# Evaluate at a point
result = rastrigin({"x0": 0.0, "x1": 0.0})
print(f"BBOB Rastrigin: {result:.4f}")
