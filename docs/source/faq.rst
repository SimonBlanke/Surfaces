.. _faq:

===
FAQ
===

Frequently asked questions about Surfaces.

General Questions
=================

What is Surfaces?
-----------------

Surfaces is a Python library providing single-objective black-box optimization
test functions for benchmarking. It includes both mathematical functions from
the optimization literature and machine learning-based test functions.

Why use test functions?
-----------------------

Test functions are essential for:

- **Algorithm development**: Test new optimization algorithms on functions
  with known properties and global optima
- **Benchmarking**: Compare different algorithms fairly using standard functions
- **Education**: Learn about optimization landscapes and algorithm behavior
- **Prototyping**: Quickly test optimization pipelines before using real objectives

How is Surfaces different from other libraries?
-----------------------------------------------

Surfaces focuses on:

1. **Unified interface**: All functions share the same API
2. **ML-based functions**: Includes realistic hyperparameter optimization landscapes
3. **scipy integration**: Built-in conversion to scipy.optimize format
4. **Flexibility**: Support for loss/score modes, multiple evaluation styles

Usage Questions
===============

How do I choose a test function?
--------------------------------

Consider these factors:

- **Dimensionality**: Start with 2D functions for visualization, scale up for complexity
- **Difficulty**: Use simple functions (Sphere) for sanity checks, harder ones (Rastrigin) for stress tests
- **Characteristics**: Multimodal (Ackley) for testing global search, unimodal (Sphere) for local search

What's the difference between loss and score?
---------------------------------------------

- **Loss**: Value to minimize (lower is better). Default behavior.
- **Score**: Value to maximize (higher is better). Use ``func.score()`` or set ``metric="score"``.

Can I use custom bounds?
------------------------

Yes, you can pass custom search spaces to many optimization libraries while
using Surfaces functions. The ``search_space()`` method provides defaults,
but you're free to use different bounds.

How do I evaluate many points efficiently?
------------------------------------------

Use batch evaluation:

.. code-block:: python

    import numpy as np
    from surfaces.test_functions.algebraic import SphereFunction

    func = SphereFunction(n_dim=3)

    # Create 100 random points
    X = np.random.uniform(-5, 5, size=(100, 3))

    # Evaluate all points
    results = func.evaluate_batch(X)

Technical Questions
===================

Why does my function return different values?
---------------------------------------------

Check these common causes:

1. **Metric mode**: Are you using loss vs score mode?
2. **Parameter order**: When using arrays, parameters are sorted alphabetically
3. **Validation**: Ensure parameter names match the expected search space

How do I disable parameter validation?
--------------------------------------

Pass ``validate=False`` to the constructor:

.. code-block:: python

    func = SphereFunction(n_dim=2, validate=False)

This can improve performance in tight optimization loops.

Can I add artificial delays?
----------------------------

Yes, use the ``DelayModifier`` to simulate expensive function evaluations:

.. code-block:: python

    from surfaces.modifiers import DelayModifier

    func = SphereFunction(
        n_dim=2,
        modifiers=[DelayModifier(delay=0.1)]  # 100ms delay per evaluation
    )

What Python versions are supported?
-----------------------------------

Surfaces supports Python |min_python| and higher.

Performance Questions
=====================

Is Surfaces fast enough for benchmarking?
-----------------------------------------

Yes. The mathematical functions are implemented efficiently using NumPy.
For the ML-based functions, the bottleneck is the model training, which
is inherent to the task.

Can I parallelize evaluations?
------------------------------

Surfaces functions are not parallelized internally, but you can use
standard parallelization tools (joblib, multiprocessing) to evaluate
multiple points in parallel.
