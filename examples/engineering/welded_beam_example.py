"""Welded Beam Design - Minimize fabrication cost of a welded beam."""

from surfaces.test_functions.algebraic import WeldedBeamFunction

welded_beam = WeldedBeamFunction()

# First evaluation
params1 = {"h": 0.5, "l": 4.0, "t": 5.0, "b": 0.5}
score1 = welded_beam(params1)
print(f"params: {params1} -> score: {score1:.4f}")

# Second evaluation
params2 = {"h": 1.0, "l": 2.0, "t": 3.0, "b": 1.0}
score2 = welded_beam(params2)
print(f"params: {params2} -> score: {score2:.4f}")
