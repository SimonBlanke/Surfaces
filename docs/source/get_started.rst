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

Installation
============

Install from PyPI:

.. code-block:: bash

    pip install surfaces

Your First Test Function
========================

Let's evaluate the classic Sphere function, one of the simplest test functions
where the global minimum is at the origin:

.. code-block:: python

    from surfaces.test_functions import SphereFunction

    # Create a 2D Sphere function
    func = SphereFunction(n_dim=2)

    # Evaluate at a point
    loss = func({"x0": 1.0, "x1": 2.0})
    print(f"f(1, 2) = {loss}")  # Output: 5.0

    # The global minimum is at (0, 0)
    optimal = func({"x0": 0.0, "x1": 0.0})
    print(f"f(0, 0) = {optimal}")  # Output: 0.0

Understanding the Interface
===========================

All Surfaces test functions share a consistent interface:

Evaluation Methods
------------------

There are multiple ways to evaluate a function:

.. code-block:: python

    from surfaces.test_functions import AckleyFunction

    func = AckleyFunction()

    # 1. Dictionary input (primary interface)
    result = func({"x0": 1.0, "x1": 2.0})

    # 2. Keyword arguments
    result = func(x0=1.0, x1=2.0)

    # 3. Positional arguments via evaluate()
    result = func.evaluate(1.0, 2.0)

    # 4. NumPy array via evaluate_array()
    import numpy as np
    result = func.evaluate_array(np.array([1.0, 2.0]))

Loss vs Score
-------------

Every function supports both minimization and maximization:

.. code-block:: python

    func = SphereFunction(n_dim=2)

    # Default: returns loss (for minimization)
    loss = func({"x0": 1.0, "x1": 1.0})  # 2.0

    # Explicit loss (always positive, lower is better)
    loss = func.loss({"x0": 1.0, "x1": 1.0})  # 2.0

    # Explicit score (negative loss, higher is better)
    score = func.score({"x0": 1.0, "x1": 1.0})  # -2.0

Search Space
------------

Every function provides a default search space:

.. code-block:: python

    func = SphereFunction(n_dim=3)

    # Get the default search space
    space = func.search_space()
    # {'x0': array([...]), 'x1': array([...]), 'x2': array([...])}

    # The space contains discrete sample points within bounds
    print(f"x0 range: [{space['x0'].min()}, {space['x0'].max()}]")

Using with scipy
================

Surfaces integrates seamlessly with scipy.optimize:

.. code-block:: python

    from surfaces.test_functions import RosenbrockFunction
    from scipy.optimize import minimize

    # Create the function
    func = RosenbrockFunction(n_dim=3)

    # Convert to scipy format
    objective, bounds, x0 = func.to_scipy()

    # Run optimization
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    print(f"Found minimum at: {result.x}")
    print(f"Minimum value: {result.fun}")

Available Functions
===================

Surfaces provides three categories of test functions:

Mathematical Functions
----------------------

Classic optimization test functions from the literature:

- **1D Functions**: Gramacy & Lee
- **2D Functions**: Ackley, Beale, Booth, Himmelblau, Rosenbrock (2D), etc.
- **N-D Functions**: Sphere, Rastrigin, Rosenbrock, Griewank, Styblinski-Tang

Machine Learning Functions
--------------------------

Test functions based on real ML model training:

- **Classification**: KNeighborsClassifierFunction
- **Regression**: KNeighborsRegressorFunction, GradientBoostingRegressorFunction

Next Steps
==========

Now that you understand the basics:

- :doc:`installation` - Detailed installation options
- :doc:`user_guide/mathematical` - Explore all mathematical functions
- :doc:`user_guide/machine_learning` - Learn about ML-based functions
- :doc:`examples` - See more code examples
- :doc:`api_reference` - Complete API documentation
