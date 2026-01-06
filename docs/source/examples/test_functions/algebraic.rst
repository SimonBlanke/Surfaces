.. _example_algebraic:

===================
Algebraic Functions
===================

Examples using classic mathematical test functions.

.. contents:: On this page
   :local:
   :depth: 2

----

Basic Algebraic Functions
=========================

.. code-block:: python

    import numpy as np
    from surfaces.test_functions.algebraic import (
        SphereFunction,
        RastriginFunction,
        RosenbrockFunction,
        AckleyFunction,
    )

    # Create functions
    sphere = SphereFunction(n_dim=5)
    rastrigin = RastriginFunction(n_dim=5)
    rosenbrock = RosenbrockFunction(n_dim=5)
    ackley = AckleyFunction()  # Fixed 2D

    # Evaluate at random points
    for func in [sphere, rastrigin, rosenbrock, ackley]:
        space = func.search_space
        sample = {k: np.random.choice(v) for k, v in space.items()}
        result = func(sample)
        print(f"{func.__class__.__name__}: {result:.4f}")

----

Comparing Function Landscapes
=============================

.. code-block:: python

    """Compare difficulty of different algebraic functions."""

    import numpy as np
    from surfaces.test_functions.algebraic import SphereFunction, RastriginFunction

    def random_search(func, n_iter=1000, seed=42):
        np.random.seed(seed)
        space = func.search_space
        best = float('inf')
        for _ in range(n_iter):
            sample = {k: np.random.choice(v) for k, v in space.items()}
            result = func(sample)
            best = min(best, result)
        return best

    sphere = SphereFunction(n_dim=10)
    rastrigin = RastriginFunction(n_dim=10)

    print("Random search (1000 iterations):")
    print(f"  Sphere: {random_search(sphere):.6f}")
    print(f"  Rastrigin: {random_search(rastrigin):.6f}")

----

Scaling with Dimension
======================

.. code-block:: python

    """Test how functions scale with dimensionality."""

    import numpy as np
    from surfaces.test_functions.algebraic import RastriginFunction
    import time

    for n_dim in [2, 5, 10, 20, 50]:
        func = RastriginFunction(n_dim=n_dim)
        space = func.search_space

        start = time.time()
        for _ in range(1000):
            sample = {k: np.random.choice(v) for k, v in space.items()}
            func(sample)
        elapsed = time.time() - start

        print(f"n_dim={n_dim:2d}: {elapsed*1000:.1f}ms for 1000 evals")
