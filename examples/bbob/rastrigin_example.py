"""BBOB Rastrigin Rotated (f15) - Multimodal with adequate global structure."""

from surfaces.test_functions.bbob import RastriginRotated

rastrigin = RastriginRotated(n_dim=2, instance=1)

# First evaluation
params1 = {"x0": 1.0, "x1": 1.0}
score1 = rastrigin(params1)
print(f"params: {params1} -> score: {score1:.4f}")

# Second evaluation
params2 = {"x0": 0.0, "x1": 0.0}
score2 = rastrigin(params2)
print(f"params: {params2} -> score: {score2:.4f}")
