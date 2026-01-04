"""Welded Beam Design - Minimize fabrication cost of a welded beam."""

from surfaces.test_functions import WeldedBeamFunction

# Create the welded beam design function
welded_beam = WeldedBeamFunction()

# Evaluate a design: h=weld thickness, l=weld length, t=beam height, b=beam width
cost = welded_beam({"h": 0.5, "l": 4.0, "t": 5.0, "b": 0.5})
print(f"Welded beam cost: {cost:.4f}")
