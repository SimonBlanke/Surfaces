"""Combining multiple modifiers for realistic optimization scenarios."""

import time

from surfaces.modifiers import DelayModifier, GaussianNoise
from surfaces.test_functions import SphereFunction

# Real-world scenario: expensive noisy function
# Modifiers are applied in order: delay first, then noise
delay = DelayModifier(delay=0.05)
noise = GaussianNoise(sigma=0.1, seed=42)

realistic_sphere = SphereFunction(n_dim=3, modifiers=[delay, noise])

print("Combined Modifiers Example")
print("=" * 40)

point = {"x0": 1.0, "x1": 1.0, "x2": 1.0}

# Multiple evaluations at same point
print(f"\nEvaluating at {point}:")
print(f"True value: {realistic_sphere.true_value(point)}")

results = []
start = time.perf_counter()
for i in range(5):
    result = realistic_sphere(point)
    results.append(result)
    print(f"  Evaluation {i+1}: {result:.4f}")
elapsed = time.perf_counter() - start

print("\nStatistics:")
print(f"  Mean: {sum(results)/len(results):.4f}")
print(f"  Total time: {elapsed*1000:.1f}ms")
print(f"  Per evaluation: {elapsed/len(results)*1000:.1f}ms")

# Access modifiers list
modifiers = realistic_sphere.modifiers
print("\nModifiers info:")
print(f"  Number of modifiers: {len(modifiers)}")
print(f"  Modifiers: {modifiers}")

# Access modifier properties directly
print(f"  Last noise value: {noise.last_noise:.4f}")
print(f"  Total noise applications: {noise.evaluation_count}")

# Reset all modifiers
print("\nResetting modifiers...")
realistic_sphere.reset_modifiers()
print(f"  Noise evaluation count after reset: {noise.evaluation_count}")
