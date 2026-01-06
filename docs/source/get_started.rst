.. _get_started:

===========
Get Started
===========

This guide walks you through your first steps with Surfaces in under 5 minutes.

What is Surfaces?
=================

Surfaces is a Python library that provides standardized test functions for
optimization algorithm benchmarking. These functions have known properties
(global optima, landscapes, difficulty) that make them ideal for:

- Testing new optimization algorithms
- Comparing algorithm performance
- Educational purposes
- Prototyping optimization pipelines

Quick Install
=============

.. code-block:: bash

    pip install surfaces

----

Why Surfaces?
=============

.. grid:: 2 2 2 2
   :gutter: 4

   .. grid-item-card:: Curated Test Functions
      :link: get_started/curated_test_functions
      :link-type: doc

      Not just a random collection. Surfaces provides carefully selected
      test functions from established benchmarks like CEC and BBOB,
      plus real-world ML and engineering problems.

   .. grid-item-card:: Plug & Play Integration
      :link: get_started/plug_and_play_integration
      :link-type: doc

      Works out of the box with scipy, Optuna, SMAC, Ray Tune,
      Gradient-Free-Optimizers, and Hyperactive. No adapters needed.

   .. grid-item-card:: Machine Learning Accelerated
      :link: get_started/machine_learning_accelerated
      :link-type: doc

      Pre-trained surrogate models let you benchmark expensive
      ML hyperparameter optimization problems instantly.

   .. grid-item-card:: Minimal Dependencies
      :link: get_started/minimal_dependencies
      :link-type: doc

      Core installation requires only numpy. Add optional features
      like ML functions or visualization only when you need them.

----

Your First Test Function
========================

Let's evaluate the classic Sphere function:

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction

    # Create a 2D Sphere function
    func = SphereFunction(n_dim=2)

    # Evaluate at a point
    loss = func({"x0": 1.0, "x1": 2.0})
    print(f"f(1, 2) = {loss}")  # Output: 5.0

    # The global minimum is at (0, 0)
    optimal = func({"x0": 0.0, "x1": 0.0})
    print(f"f(0, 0) = {optimal}")  # Output: 0.0

----

Next Steps
==========

- :doc:`installation` - Detailed installation options
- :doc:`user_guide` - In-depth tutorials and explanations
- :doc:`examples` - Code examples and use cases
- :doc:`api_reference` - Complete API documentation


.. toctree::
   :maxdepth: 1
   :hidden:

   get_started/curated_test_functions
   get_started/plug_and_play_integration
   get_started/machine_learning_accelerated
   get_started/minimal_dependencies
