"""CEC 2013 Sphere - Unimodal baseline function."""

from surfaces.test_functions.cec.cec2013 import Sphere

sphere = Sphere(n_dim=10)

# First evaluation
params1 = {f"x{i}": 1.0 for i in range(10)}
score1 = sphere(params1)
print(f"params: x0=1.0, ..., x9=1.0 -> score: {score1:.4f}")

# Second evaluation
params2 = {f"x{i}": 0.0 for i in range(10)}
score2 = sphere(params2)
print(f"params: x0=0.0, ..., x9=0.0 -> score: {score2:.4f}")
