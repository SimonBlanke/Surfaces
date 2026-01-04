"""Rastrigin Function - A highly multimodal benchmark function."""

from surfaces.test_functions import RastriginFunction

# Create a 2-dimensional Rastrigin function
rastrigin = RastriginFunction(n_dim=2)

# Global minimum is at the origin
result = rastrigin({"x0": 0.0, "x1": 0.0})
print(f"Rastrigin(0, 0) = {result}")  # Output: 0.0
