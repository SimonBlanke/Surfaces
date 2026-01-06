"""Ackley Function - Multimodal with a nearly flat outer region."""

from surfaces.test_functions.algebraic import AckleyFunction

ackley = AckleyFunction()

# First evaluation
params1 = {"x0": 2.0, "x1": 2.0}
score1 = ackley(params1)
print(f"params: {params1} -> score: {score1:.4f}")

# Second evaluation (global optimum)
params2 = {"x0": 0.0, "x1": 0.0}
score2 = ackley(params2)
print(f"params: {params2} -> score: {score2:.4f}")
