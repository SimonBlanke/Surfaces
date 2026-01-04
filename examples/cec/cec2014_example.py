"""CEC 2014 Rotated Ackley - Multimodal benchmark function."""

from surfaces.test_functions.cec.cec2014 import ShiftedRotatedAckley

# Create a 10-dimensional CEC 2014 Ackley function
ackley = ShiftedRotatedAckley(n_dim=10)

# Evaluate at a point
origin = {f"x{i}": 0.0 for i in range(10)}
result = ackley(origin)
print(f"CEC 2014 Ackley: {result:.4f}")
