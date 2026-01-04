"""CEC 2013 Sphere - Unimodal baseline function."""

from surfaces.test_functions.cec.cec2013 import Sphere

# Create a 10-dimensional CEC 2013 Sphere function
sphere = Sphere(n_dim=10)

# Evaluate at a point
origin = {f"x{i}": 0.0 for i in range(10)}
result = sphere(origin)
print(f"CEC 2013 Sphere: {result:.4f}")
