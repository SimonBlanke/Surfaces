.. _user_guide_algebraic_nd:

============
N-D Functions
============

N-dimensional (scalable) test functions work in any number of dimensions.
They are essential for testing how optimizers scale with problem size.

----

Available N-D Functions
=======================

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Function
     - Type
     - Characteristics
   * - ``SphereFunction``
     - Unimodal
     - Simplest test function, convex
   * - ``RastriginFunction``
     - Multimodal
     - Highly multimodal, separable
   * - ``RosenbrockFunction``
     - Unimodal
     - Narrow valley, non-separable
   * - ``GriewankFunction``
     - Multimodal
     - Product term creates local minima
   * - ``StyblinskiTangFunction``
     - Multimodal
     - Asymmetric landscape

----

Sphere Function
===============

The simplest and most fundamental test function. Convex, separable,
unimodal. If your optimizer cannot solve Sphere, something is wrong.

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction

    # Any dimension
    func = SphereFunction(n_dim=10)

    # Global minimum at origin
    result = func({f"x{i}": 0.0 for i in range(10)})  # 0.0

**Properties:**

- Formula: f(x) = sum(x_i^2)
- Global minimum: f(0, ..., 0) = 0
- Convex, separable, unimodal
- Difficulty increases with dimension

Rastrigin Function
==================

A highly multimodal function with a regular pattern of local minima.

.. code-block:: python

    from surfaces.test_functions.algebraic import RastriginFunction

    func = RastriginFunction(n_dim=10)

    # Global minimum at origin
    result = func({f"x{i}": 0.0 for i in range(10)})  # 0.0

**Properties:**

- Formula: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
- Global minimum: f(0, ..., 0) = 0
- ~10^n local minima in n dimensions
- Separable but highly multimodal

Rosenbrock Function
===================

A non-separable function with a narrow, curved valley leading to
the global minimum.

.. code-block:: python

    from surfaces.test_functions.algebraic import RosenbrockFunction

    func = RosenbrockFunction(n_dim=10)

    # Global minimum at (1, 1, ..., 1)
    result = func({f"x{i}": 1.0 for i in range(10)})  # 0.0

**Properties:**

- Formula: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
- Global minimum: f(1, ..., 1) = 0
- Non-separable (adjacent variables interact)
- Tests ability to follow narrow valleys

Griewank Function
=================

Combines a sum term with a product term that creates local minima.

.. code-block:: python

    from surfaces.test_functions.algebraic import GriewankFunction

    func = GriewankFunction(n_dim=10)
    result = func({f"x{i}": 0.0 for i in range(10)})  # 0.0

**Properties:**

- Global minimum: f(0, ..., 0) = 0
- Number of local minima increases with dimension
- Product term becomes less significant at high dimensions

Styblinski-Tang Function
========================

An asymmetric multimodal function.

.. code-block:: python

    from surfaces.test_functions.algebraic import StyblinskiTangFunction

    func = StyblinskiTangFunction(n_dim=10)

    # Global minimum at x_i = -2.903534
    x_opt = {f"x{i}": -2.903534 for i in range(10)}
    result = func(x_opt)

**Properties:**

- Global minimum: x_i = -2.903534
- Minimum value: f(x*) = -39.16599 * n
- Asymmetric landscape

----

Scaling Behavior
================

N-D functions are crucial for understanding how optimizers scale:

.. code-block:: python

    from surfaces.test_functions.algebraic import RastriginFunction
    import time

    for n_dim in [2, 5, 10, 20, 50, 100]:
        func = RastriginFunction(n_dim=n_dim)

        # Measure evaluation time
        start = time.time()
        for _ in range(1000):
            func(func.search_space_sample())
        elapsed = time.time() - start

        print(f"n_dim={n_dim:3d}: {elapsed:.3f}s for 1000 evaluations")

----

Choosing Dimensions
===================

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Dimension
     - Use Case
   * - 2-3
     - Visualization, debugging
   * - 5-10
     - Standard benchmarking
   * - 20-50
     - Testing scaling behavior
   * - 100+
     - High-dimensional optimization research

----

Next Steps
==========

- :doc:`/user_guide/test_functions/bbob` - BBOB benchmark suite
- :doc:`/user_guide/test_functions/cec` - CEC competition functions
- :doc:`/api_reference/test_functions/algebraic` - Complete API reference
