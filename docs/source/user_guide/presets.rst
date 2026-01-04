.. _user_guide_presets:

=======
Presets
=======

Presets are pre-configured collections of test functions for common
benchmarking scenarios. They save time by providing ready-to-use
function sets.

----

Why Use Presets?
================

Instead of manually selecting functions:

.. code-block:: python

    # Manual selection - tedious
    from surfaces.test_functions import (
        SphereFunction,
        RastriginFunction,
        RosenbrockFunction,
        AckleyFunction,
        # ... many more
    )

    functions = [
        SphereFunction(n_dim=10),
        RastriginFunction(n_dim=10),
        # ...
    ]

Use a preset:

.. code-block:: python

    # With preset - simple
    from surfaces.presets import classic_benchmark

    functions = classic_benchmark(n_dim=10)

----

Available Presets
=================

Classic Benchmark
-----------------

Standard test functions commonly used in optimization papers.

.. code-block:: python

    from surfaces.presets import classic_benchmark

    functions = classic_benchmark(n_dim=10)

    for func in functions:
        result = func(func.search_space_sample())
        print(f"{func.__class__.__name__}: {result:.4f}")

**Includes:**

- Sphere
- Rastrigin
- Rosenbrock
- Ackley
- Griewank
- Styblinski-Tang

Quick Test
----------

A small set for fast iteration during development.

.. code-block:: python

    from surfaces.presets import quick_test

    functions = quick_test(n_dim=5)

**Includes:**

- Sphere (easy)
- Rastrigin (medium)
- Rosenbrock (hard)

Multimodal Challenge
--------------------

Functions with many local optima to test global optimization.

.. code-block:: python

    from surfaces.presets import multimodal_challenge

    functions = multimodal_challenge(n_dim=10)

**Includes:**

- Rastrigin
- Ackley
- Griewank
- Schwefel
- Levy

----

Using Presets for Benchmarking
==============================

.. code-block:: python

    from surfaces.presets import classic_benchmark
    from gradient_free_optimizers import BayesianOptimizer

    functions = classic_benchmark(n_dim=10)
    results = {}

    for func in functions:
        opt = BayesianOptimizer(func.search_space())
        opt.search(func, n_iter=100)

        results[func.__class__.__name__] = opt.best_score

    # Print results table
    for name, score in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:25s}: {score:.6f}")

----

Creating Custom Presets
=======================

You can create your own presets:

.. code-block:: python

    from surfaces.test_functions import (
        SphereFunction,
        RastriginFunction,
        RosenbrockFunction,
    )

    def my_preset(n_dim=10):
        return [
            SphereFunction(n_dim=n_dim),
            RastriginFunction(n_dim=n_dim),
            RosenbrockFunction(n_dim=n_dim),
        ]

    # Use your preset
    for func in my_preset(n_dim=20):
        print(func.__class__.__name__)

----

Preset Parameters
=================

All presets accept common parameters:

.. code-block:: python

    functions = classic_benchmark(
        n_dim=10,           # Number of dimensions
        # Additional parameters may vary by preset
    )

----

Combining Presets
=================

Combine multiple presets:

.. code-block:: python

    from surfaces.presets import classic_benchmark, multimodal_challenge

    all_functions = (
        classic_benchmark(n_dim=10) +
        multimodal_challenge(n_dim=10)
    )

    # Remove duplicates if needed
    unique_functions = list({type(f): f for f in all_functions}.values())

----

Next Steps
==========

- :doc:`test_functions/index` - All available test functions
- :doc:`integrations/index` - Use presets with optimization frameworks
- :doc:`/api_reference/presets` - Complete presets API
