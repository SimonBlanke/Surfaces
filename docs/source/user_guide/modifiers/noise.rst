.. _user_guide_noise:

=====
Noise
=====

Noise modifiers add stochastic variance to function evaluations.
They test how optimizers handle noisy objective functions.

----

Why Add Noise?
==============

Real-world optimization problems often have noisy evaluations:

- **Machine learning**: Cross-validation variance
- **Simulations**: Monte Carlo sampling
- **Physical experiments**: Measurement uncertainty
- **A/B testing**: Statistical fluctuations

Testing on noisy functions reveals:

- How many evaluations your optimizer needs
- Whether it handles variance correctly
- If it converges to the true optimum or gets stuck

----

Available Noise Types
=====================

Gaussian Noise
--------------

Adds normally distributed noise to the objective value.

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction
    from surfaces.modifiers import GaussianNoise

    # Create function with Gaussian noise (standard deviation 0.1)
    noisy = SphereFunction(
        n_dim=3,
        modifiers=[GaussianNoise(sigma=0.1)]
    )

    # Multiple evaluations at same point give different results
    point = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
    results = [noisy(point) for _ in range(10)]
    print(f"Mean: {sum(results)/len(results):.3f}")
    print(f"Std:  {(sum((r-sum(results)/len(results))**2 for r in results)/len(results))**0.5:.3f}")

**Parameters:**

- ``sigma``: Standard deviation of the noise

Uniform Noise
-------------

Adds uniformly distributed noise.

.. code-block:: python

    from surfaces.modifiers import UniformNoise

    noisy = SphereFunction(
        n_dim=3,
        modifiers=[UniformNoise(low=-0.1, high=0.1)]
    )

**Parameters:**

- ``low``: Lower bound of uniform distribution
- ``high``: Upper bound of uniform distribution

----

Noise Levels
============

The noise level significantly affects optimization difficulty:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Sigma
     - Difficulty
     - Use Case
   * - 0.01
     - Low
     - Most optimizers handle this well
   * - 0.1
     - Medium
     - Requires noise-tolerant algorithms
   * - 1.0
     - High
     - Very challenging, needs many samples

----

Signal-to-Noise Ratio
=====================

Consider the noise level relative to the function's range:

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction
    from surfaces.modifiers import GaussianNoise

    base = SphereFunction(n_dim=3)

    # Get function range
    space = base.search_space()
    samples = [base({f"x{i}": space[f"x{i}"][j] for i in range(3)})
               for j in range(100)]
    func_range = max(samples) - min(samples)

    # Choose noise relative to range
    sigma = 0.01 * func_range  # 1% noise
    noisy = SphereFunction(
        n_dim=3,
        modifiers=[GaussianNoise(sigma=sigma)]
    )

----

Reproducibility
===============

For reproducible benchmarks, set a random seed:

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction
    from surfaces.modifiers import GaussianNoise

    # Use seed parameter for reproducibility
    noisy = SphereFunction(
        n_dim=3,
        modifiers=[GaussianNoise(sigma=0.1, seed=42)]
    )

    point = {"x0": 1.0, "x1": 1.0, "x2": 1.0}

    # Create another function with same seed
    noisy2 = SphereFunction(
        n_dim=3,
        modifiers=[GaussianNoise(sigma=0.1, seed=42)]
    )

    # Results are reproducible across instances with same seed
    result1 = noisy(point)
    result2 = noisy2(point)
    assert result1 == result2

----

Optimizer Strategies
====================

When optimizing noisy functions:

1. **Multiple evaluations**: Average several evaluations at each point
2. **Larger populations**: Use population-based methods
3. **Surrogate models**: Build noise-robust surrogate approximations
4. **Statistical tests**: Use proper comparison methods

.. code-block:: python

    # Example: Averaging multiple evaluations
    def robust_evaluate(func, params, n_samples=5):
        results = [func(params) for _ in range(n_samples)]
        return sum(results) / len(results)

    avg_result = robust_evaluate(noisy, point, n_samples=10)

----

Next Steps
==========

- :doc:`/user_guide/test_functions/index` - Test functions to apply noise to
- :doc:`/api_reference/noise` - Complete noise API reference
