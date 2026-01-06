"""Using delay modifiers to simulate expensive function evaluations."""

import time

from surfaces.modifiers import DelayModifier
from surfaces.test_functions import SphereFunction

# Create a function with artificial delay
slow_sphere = SphereFunction(
    n_dim=2,
    modifiers=[DelayModifier(delay=0.1)],  # 100ms delay per evaluation
)

point = {"x0": 1.0, "x1": 1.0}

print("DelayModifier Example")
print("=" * 40)

# Time a single evaluation
start = time.perf_counter()
result = slow_sphere(point)
elapsed = time.perf_counter() - start

print("Single evaluation:")
print(f"  Result: {result}")
print(f"  Time: {elapsed*1000:.1f}ms (expected: ~100ms)")

# Time multiple evaluations
n_evals = 5
start = time.perf_counter()
for _ in range(n_evals):
    slow_sphere(point)
elapsed = time.perf_counter() - start

print(f"\n{n_evals} evaluations:")
print(f"  Total time: {elapsed*1000:.1f}ms (expected: ~{n_evals*100}ms)")
print(f"  Per evaluation: {elapsed/n_evals*1000:.1f}ms")

# true_value() bypasses modifiers - useful for comparison
print("\nBypassing delay with true_value():")
start = time.perf_counter()
true_result = slow_sphere.true_value(point)
elapsed = time.perf_counter() - start
print(f"  Result: {true_result}")
print(f"  Time: {elapsed*1000:.2f}ms (no delay)")
