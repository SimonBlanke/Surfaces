"""CEC 2017 Rastrigin - Shifted and rotated multimodal function."""

from surfaces.test_functions.cec.cec2017 import ShiftedRotatedRastrigin

# Create a 10-dimensional CEC 2017 Rastrigin function
rastrigin = ShiftedRotatedRastrigin(n_dim=10)

# Evaluate at a point
origin = {f"x{i}": 0.0 for i in range(10)}
result = rastrigin(origin)
print(f"CEC 2017 Rastrigin: {result:.4f}")
