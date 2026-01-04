"""BBOB Ellipsoidal Rotated (f10) - Ill-conditioned unimodal function."""

from surfaces.test_functions.bbob import EllipsoidalRotated

# Create a 3-dimensional ill-conditioned ellipsoidal function
ellipsoid = EllipsoidalRotated(n_dim=3, instance=1)

# Evaluate at a point
result = ellipsoid({"x0": 0.0, "x1": 0.0, "x2": 0.0})
print(f"BBOB Ellipsoidal: {result:.4f}")
