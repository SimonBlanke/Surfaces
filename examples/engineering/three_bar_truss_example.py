"""Three-Bar Truss Design - Minimize volume of a truss structure."""

from surfaces.test_functions.algebraic import ThreeBarTrussFunction

truss = ThreeBarTrussFunction()

# First evaluation
params1 = {"A1": 0.5, "A2": 0.5}
score1 = truss(params1)
print(f"params: {params1} -> score: {score1:.4f}")

# Second evaluation
params2 = {"A1": 0.8, "A2": 0.4}
score2 = truss(params2)
print(f"params: {params2} -> score: {score2:.4f}")
