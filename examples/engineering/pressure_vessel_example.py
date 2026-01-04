"""Pressure Vessel Design - Minimize cost of a cylindrical vessel."""

from surfaces.test_functions import PressureVesselFunction

# Create the pressure vessel design function
pressure_vessel = PressureVesselFunction()

# Evaluate a design: Ts=shell thickness, Th=head thickness, R=radius, L=length
cost = pressure_vessel({"Ts": 1.0, "Th": 1.0, "R": 50.0, "L": 100.0})
print(f"Pressure vessel cost: {cost:.4f}")
