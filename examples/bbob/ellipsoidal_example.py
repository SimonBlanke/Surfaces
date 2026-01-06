"""BBOB Ellipsoidal Rotated (f10) - Ill-conditioned unimodal function."""

from surfaces.test_functions.benchmark.bbob import EllipsoidalRotated

ellipsoid = EllipsoidalRotated(n_dim=3, instance=1)

# First evaluation
params1 = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
score1 = ellipsoid(params1)
print(f"params: {params1} -> score: {score1:.4f}")

# Second evaluation
params2 = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
score2 = ellipsoid(params2)
print(f"params: {params2} -> score: {score2:.4f}")
