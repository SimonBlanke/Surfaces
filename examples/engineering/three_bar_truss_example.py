"""Three-Bar Truss Design - Minimize volume of a truss structure."""

from surfaces.test_functions import ThreeBarTrussFunction

# Create the three-bar truss design function
truss = ThreeBarTrussFunction()

# Evaluate a design: A1 and A2 are cross-sectional areas
volume = truss({"A1": 0.5, "A2": 0.5})
print(f"Truss volume: {volume:.4f}")
