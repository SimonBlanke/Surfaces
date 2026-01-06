"""Rastrigin Function - A highly multimodal benchmark function."""

from surfaces.test_functions.algebraic import RastriginFunction

rastrigin = RastriginFunction(n_dim=2)

# First evaluation
params1 = {"x0": 1.0, "x1": 1.0}
score1 = rastrigin(params1)
print(f"params: {params1} -> score: {score1}")

# Second evaluation (global optimum)
params2 = {"x0": 0.0, "x1": 0.0}
score2 = rastrigin(params2)
print(f"params: {params2} -> score: {score2}")
