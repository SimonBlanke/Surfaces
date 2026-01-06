"""Using noise modifiers to simulate noisy objective functions."""

from surfaces.test_functions import SphereFunction
from surfaces.modifiers import GaussianNoise, UniformNoise, MultiplicativeNoise

# Create a function with Gaussian noise
gaussian_noise = GaussianNoise(sigma=0.1, seed=42)
noisy_sphere = SphereFunction(n_dim=2, modifiers=[gaussian_noise])

# Evaluate at the same point multiple times - results vary due to noise
point = {"x0": 1.0, "x1": 1.0}
print("Gaussian Noise (sigma=0.1):")
print(f"  True value at {point}: {noisy_sphere.true_value(point)}")
for i in range(5):
    result = noisy_sphere(point)
    print(f"  Evaluation {i+1}: {result:.4f}")

# Access the noise modifier to see last noise value
print(f"  Last noise added: {gaussian_noise.last_noise:.4f}")

# Uniform noise example
uniform_sphere = SphereFunction(
    n_dim=2,
    modifiers=[UniformNoise(low=-0.5, high=0.5, seed=42)]
)

print("\nUniform Noise (low=-0.5, high=0.5):")
print(f"  True value at {point}: {uniform_sphere.true_value(point)}")
for i in range(5):
    result = uniform_sphere(point)
    print(f"  Evaluation {i+1}: {result:.4f}")

# Multiplicative noise - scales with function value
mult_sphere = SphereFunction(
    n_dim=2,
    modifiers=[MultiplicativeNoise(sigma=0.1, seed=42)]
)

print("\nMultiplicative Noise (sigma=0.1):")
print(f"  True value at {point}: {mult_sphere.true_value(point)}")
for i in range(5):
    result = mult_sphere(point)
    print(f"  Evaluation {i+1}: {result:.4f}")

# Noise scheduling - decay noise over evaluations
scheduled_noise = GaussianNoise(
    sigma=1.0,
    sigma_final=0.01,
    schedule="linear",
    total_evaluations=100,
    seed=42
)
scheduled_sphere = SphereFunction(n_dim=2, modifiers=[scheduled_noise])

print("\nScheduled Noise (linear decay from 1.0 to 0.01 over 100 evals):")
for eval_num in [1, 25, 50, 75, 100]:
    # Reset to start fresh
    scheduled_noise = GaussianNoise(
        sigma=1.0,
        sigma_final=0.01,
        schedule="linear",
        total_evaluations=100,
        seed=42
    )
    # Simulate evaluations up to eval_num
    for _ in range(eval_num):
        scheduled_noise.apply(0.0, {}, {})
    print(f"  After {eval_num} evals, effective sigma: {scheduled_noise.sigma:.4f}")
