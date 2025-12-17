.. _examples_basic_usage:

===========
Basic Usage
===========

Basic examples of using Surfaces test functions.

Evaluating a Simple Function
============================

The Sphere function is the simplest test function:

.. code-block:: python

    from surfaces.test_functions import SphereFunction

    # Create a 2D Sphere function
    func = SphereFunction(n_dim=2)

    # Evaluate at a point
    result = func({"x0": 1.0, "x1": 2.0})
    print(f"f(1, 2) = {result}")  # 5.0

    # The minimum is at the origin
    result = func({"x0": 0.0, "x1": 0.0})
    print(f"f(0, 0) = {result}")  # 0.0

Using Different Input Formats
=============================

Surfaces supports multiple input formats:

.. code-block:: python

    from surfaces.test_functions import AckleyFunction
    import numpy as np

    func = AckleyFunction()

    # Dictionary input
    r1 = func({"x0": 1.0, "x1": 2.0})

    # Keyword arguments
    r2 = func(x0=1.0, x1=2.0)

    # Positional arguments
    r3 = func.evaluate(1.0, 2.0)

    # NumPy array
    r4 = func.evaluate_array(np.array([1.0, 2.0]))

    # All return the same value
    print(f"All equal: {r1 == r2 == r3 == r4}")

Batch Evaluation
================

Evaluate multiple points efficiently:

.. code-block:: python

    import numpy as np
    from surfaces.test_functions import RastriginFunction

    func = RastriginFunction(n_dim=3)

    # Generate 100 random points
    np.random.seed(42)
    X = np.random.uniform(-5, 5, size=(100, 3))

    # Evaluate all points
    results = func.evaluate_batch(X)

    print(f"Shape: {results.shape}")  # (100,)
    print(f"Min value: {results.min():.4f}")
    print(f"Max value: {results.max():.4f}")

Loss vs Score Mode
==================

Every function supports both minimization and maximization:

.. code-block:: python

    from surfaces.test_functions import SphereFunction

    # Loss mode (default): lower is better
    func_loss = SphereFunction(n_dim=2, metric="loss")
    loss = func_loss({"x0": 1.0, "x1": 1.0})
    print(f"Loss: {loss}")  # 2.0

    # Score mode: higher is better
    func_score = SphereFunction(n_dim=2, metric="score")
    score = func_score({"x0": 1.0, "x1": 1.0})
    print(f"Score: {score}")  # -2.0

    # Explicit methods work regardless of mode
    func = SphereFunction(n_dim=2)
    print(f"Explicit loss: {func.loss(x0=1.0, x1=1.0)}")   # 2.0
    print(f"Explicit score: {func.score(x0=1.0, x1=1.0)}")  # -2.0

Working with Search Spaces
==========================

Get the default search space:

.. code-block:: python

    from surfaces.test_functions import RosenbrockFunction

    func = RosenbrockFunction(n_dim=3)

    # Get the search space
    space = func.search_space()

    print(f"Parameters: {list(space.keys())}")
    print(f"x0 has {len(space['x0'])} sample points")
    print(f"x0 range: [{space['x0'].min()}, {space['x0'].max()}]")

    # Get continuous bounds
    lower, upper = func.get_bounds()
    print(f"Lower bounds: {lower}")
    print(f"Upper bounds: {upper}")

Using N-Dimensional Functions
=============================

Scale functions to any dimension:

.. code-block:: python

    from surfaces.test_functions import SphereFunction

    # Create functions with different dimensions
    for n_dim in [2, 5, 10, 100]:
        func = SphereFunction(n_dim=n_dim)

        # Create params at origin
        params = {f"x{i}": 0.0 for i in range(n_dim)}

        # All return 0 at origin
        result = func(params)
        print(f"Sphere({n_dim}D) at origin = {result}")

Simulating Expensive Evaluations
================================

Add artificial delays to simulate expensive functions:

.. code-block:: python

    import time
    from surfaces.test_functions import SphereFunction

    # Fast function
    func_fast = SphereFunction(n_dim=2, sleep=0)

    start = time.time()
    for _ in range(100):
        func_fast({"x0": 0, "x1": 0})
    print(f"Fast: {time.time() - start:.3f}s for 100 evals")

    # Slow function (100ms per eval)
    func_slow = SphereFunction(n_dim=2, sleep=0.1)

    start = time.time()
    for _ in range(10):
        func_slow({"x0": 0, "x1": 0})
    print(f"Slow: {time.time() - start:.3f}s for 10 evals")
