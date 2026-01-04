"""Ackley Function - Multimodal with a nearly flat outer region."""

from surfaces.test_functions import AckleyFunction

# Create a 2-dimensional Ackley function
ackley = AckleyFunction()

# Global minimum is at the origin
result = ackley({"x0": 0.0, "x1": 0.0})
print(f"Ackley(0, 0) = {result}")  # Output: 0.0
