"""CEC 2014 Rotated Ackley - Multimodal benchmark function."""

from surfaces.test_functions.benchmark.cec.cec2014 import ShiftedRotatedAckley

ackley = ShiftedRotatedAckley(n_dim=10)

# First evaluation
params1 = {f"x{i}": 1.0 for i in range(10)}
score1 = ackley(params1)
print(f"params: x0=1.0, ..., x9=1.0 -> score: {score1:.4f}")

# Second evaluation
params2 = {f"x{i}": 0.0 for i in range(10)}
score2 = ackley(params2)
print(f"params: x0=0.0, ..., x9=0.0 -> score: {score2:.4f}")
