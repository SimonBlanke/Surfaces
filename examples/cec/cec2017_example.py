"""CEC 2017 Rastrigin - Shifted and rotated multimodal function."""

from surfaces.test_functions.benchmark.cec.cec2017 import ShiftedRotatedRastrigin

rastrigin = ShiftedRotatedRastrigin(n_dim=10)

# First evaluation
params1 = {f"x{i}": 1.0 for i in range(10)}
score1 = rastrigin(params1)
print(f"params: x0=1.0, ..., x9=1.0 -> score: {score1:.4f}")

# Second evaluation
params2 = {f"x{i}": 0.0 for i in range(10)}
score2 = rastrigin(params2)
print(f"params: x0=0.0, ..., x9=0.0 -> score: {score2:.4f}")
