"""Pressure Vessel Design - Minimize cost of a cylindrical vessel."""

from surfaces.test_functions.algebraic import PressureVesselFunction

pressure_vessel = PressureVesselFunction()

# First evaluation
params1 = {"Ts": 1.0, "Th": 1.0, "R": 50.0, "L": 100.0}
score1 = pressure_vessel(params1)
print(f"params: {params1} -> score: {score1:.4f}")

# Second evaluation
params2 = {"Ts": 0.5, "Th": 0.5, "R": 40.0, "L": 150.0}
score2 = pressure_vessel(params2)
print(f"params: {params2} -> score: {score2:.4f}")
