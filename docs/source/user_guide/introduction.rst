.. _user_guide_introduction:

============
Introduction
============

This guide covers the core concepts and interface of Surfaces test functions.

What is a Test Function?
========================

A test function (or benchmark function) is a mathematical function with known
properties used to evaluate optimization algorithms. Good test functions have:

- **Known global optimum**: You know the best possible solution
- **Defined characteristics**: Multimodal, convex, separable, etc.
- **Controllable difficulty**: Different functions pose different challenges

Surfaces provides a collection of such functions with a unified Python interface.

The BaseTestFunction Interface
==============================

All test functions in Surfaces inherit from ``BaseTestFunction`` and share
a consistent interface:

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction

    # Create a function instance
    func = SphereFunction(n_dim=3, metric="loss", sleep=0)

    # Evaluate the function
    result = func({"x0": 1.0, "x1": 2.0, "x2": 3.0})

    # Get the search space
    space = func.search_space()

Constructor Parameters
----------------------

All test functions accept these common parameters:

``metric`` : str, default="loss"
    Controls the return value:

    - ``"loss"``: Returns value to minimize (lower is better)
    - ``"score"``: Returns value to maximize (higher is better)

``sleep`` : float, default=0
    Artificial delay in seconds added to each evaluation. Useful for
    simulating expensive function evaluations.

``validate`` : bool, default=True
    Whether to validate parameters against the expected search space.
    Disable for better performance in optimization loops.

Evaluation Methods
==================

Surfaces provides multiple ways to evaluate a function:

Dictionary Interface (Primary)
------------------------------

The main interface uses dictionaries:

.. code-block:: python

    # Pass a dictionary
    result = func({"x0": 1.0, "x1": 2.0})

    # Or use keyword arguments
    result = func(x0=1.0, x1=2.0)

    # Or combine both
    result = func({"x0": 1.0}, x1=2.0)

Positional Arguments
--------------------

For convenience, you can use positional arguments:

.. code-block:: python

    # Arguments are mapped to sorted parameter names
    result = func.evaluate(1.0, 2.0)  # x0=1.0, x1=2.0

NumPy Arrays
------------

For scipy integration:

.. code-block:: python

    import numpy as np

    # Single point
    result = func.evaluate_array(np.array([1.0, 2.0]))

    # Multiple points (batch)
    X = np.array([[1, 2], [3, 4], [5, 6]])
    results = func.evaluate_batch(X)

Loss vs Score
=============

Every function can return either a loss (for minimization) or a score
(for maximization):

Using the metric Parameter
--------------------------

.. code-block:: python

    # Default: loss mode
    func_loss = SphereFunction(n_dim=2, metric="loss")
    result = func_loss({"x0": 1.0, "x1": 1.0})  # Returns 2.0

    # Score mode
    func_score = SphereFunction(n_dim=2, metric="score")
    result = func_score({"x0": 1.0, "x1": 1.0})  # Returns -2.0

Explicit Methods
----------------

You can also use explicit methods regardless of the metric setting:

.. code-block:: python

    func = SphereFunction(n_dim=2)

    # Always returns loss (value to minimize)
    loss = func.loss({"x0": 1.0, "x1": 1.0})  # 2.0

    # Always returns score (value to maximize)
    score = func.score({"x0": 1.0, "x1": 1.0})  # -2.0

Search Space
============

Every function provides a default search space via the ``search_space()`` method:

.. code-block:: python

    func = SphereFunction(n_dim=3)
    space = func.search_space()

    # space is a dictionary of parameter name -> array of values
    print(space.keys())  # dict_keys(['x0', 'x1', 'x2'])
    print(space['x0'][:5])  # Array of sample points

The search space contains discrete sample points within the function's
default bounds. This format is compatible with grid search and other
discrete optimization methods.

For continuous bounds, use ``get_bounds()``:

.. code-block:: python

    lower, upper = func.get_bounds()
    print(f"x0 range: [{lower[0]}, {upper[0]}]")

Parameter Validation
====================

By default, Surfaces validates that:

1. All required parameters are provided
2. No unexpected parameters are passed

.. code-block:: python

    func = SphereFunction(n_dim=2)

    # Missing parameter - raises ValueError
    func({"x0": 1.0})  # Error: Missing 'x1'

    # Extra parameter - raises ValueError
    func({"x0": 1.0, "x1": 2.0, "x2": 3.0})  # Error: Unexpected 'x2'

For performance-critical code, disable validation:

.. code-block:: python

    func = SphereFunction(n_dim=2, validate=False)
    result = func({"x0": 1.0, "x1": 2.0})  # No validation overhead

Artificial Delays
=================

The ``sleep`` parameter adds artificial delays to simulate expensive
evaluations. This is useful for testing parallelization strategies:

.. code-block:: python

    import time

    # No delay
    func_fast = SphereFunction(n_dim=2, sleep=0)
    start = time.time()
    func_fast({"x0": 0, "x1": 0})
    print(f"Fast: {time.time() - start:.3f}s")  # ~0.000s

    # 100ms delay
    func_slow = SphereFunction(n_dim=2, sleep=0.1)
    start = time.time()
    func_slow({"x0": 0, "x1": 0})
    print(f"Slow: {time.time() - start:.3f}s")  # ~0.100s

Next Steps
==========

- :doc:`mathematical` - Explore mathematical test functions
- :doc:`machine_learning` - Learn about ML-based functions
- :doc:`scipy_integration` - Use with scipy.optimize
