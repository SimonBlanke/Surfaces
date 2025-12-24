.. _user_guide_mathematical:

====================
Algebraic Functions
====================

Surfaces provides classic mathematical optimization test functions from
the literature. These functions have well-known properties, analytical
formulas, and global optima.

Overview
========

Algebraic functions are organized by dimensionality:

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Category
     - Count
     - Description
   * - 1D Functions
     - |n_1d|
     - Single-variable functions
   * - 2D Functions
     - |n_2d|
     - Fixed two-dimensional functions
   * - N-D Functions
     - |n_nd|
     - Scalable to any dimension

1D Functions
============

These functions operate on a single variable.

.. include:: /_generated/catalogs/algebraic_1d.rst
   :start-after: Single-variable

Example Usage
-------------

.. code-block:: python

    from surfaces.test_functions import GramacyAndLeeFunction

    func = GramacyAndLeeFunction()
    result = func({"x0": 0.5})

    # Get search space
    space = func.search_space()

.. _2d_functions:

2D Functions
============

Fixed two-dimensional functions, ideal for visualization and testing.

.. tip::

   See the :ref:`function_gallery` for visual thumbnails of all 2D functions.

.. include:: /_generated/catalogs/algebraic_2d.rst
   :start-after: Two-dimensional

Example: Ackley Function
------------------------

A widely used multimodal function with a nearly flat outer region
and a large hole at the center:

.. code-block:: python

    from surfaces.test_functions import AckleyFunction

    func = AckleyFunction()

    # Global minimum at (0, 0) with f(0, 0) = 0
    result = func({"x0": 0.0, "x1": 0.0})  # 0.0

**Properties**:

- Global minimum: f(0, 0) = 0
- Many local minima
- Difficult for gradient-based methods

Example: Rosenbrock Function (2D)
---------------------------------

The classic "banana function" with a narrow curved valley:

.. code-block:: python

    from surfaces.test_functions import RosenbrockFunction

    func = RosenbrockFunction(n_dim=2)

    # Global minimum at (1, 1) with f(1, 1) = 0
    result = func({"x0": 1.0, "x1": 1.0})  # 0.0

**Properties**:

- Global minimum: f(1, 1, ..., 1) = 0
- Narrow curved valley
- Easy to find the valley, hard to converge to minimum

.. _nd_functions:

N-Dimensional Functions
=======================

These functions scale to any number of dimensions. Specify the
dimensionality when creating the function.

.. include:: /_generated/catalogs/algebraic_nd.rst
   :start-after: Scalable

Example: Sphere Function
------------------------

The simplest test function; a convex paraboloid:

.. code-block:: python

    from surfaces.test_functions import SphereFunction

    # Create with any dimension
    func = SphereFunction(n_dim=10)

    # Global minimum at origin
    params = {f"x{i}": 0.0 for i in range(10)}
    result = func(params)  # 0.0

**Properties**:

- Global minimum: f(0, 0, ..., 0) = 0
- Convex, unimodal
- Good for sanity checking optimizers

Example: Rastrigin Function
---------------------------

Highly multimodal with regularly distributed local minima:

.. code-block:: python

    from surfaces.test_functions import RastriginFunction

    func = RastriginFunction(n_dim=5)

    # Global minimum at origin
    params = {f"x{i}": 0.0 for i in range(5)}
    result = func(params)  # 0.0

**Properties**:

- Global minimum: f(0, 0, ..., 0) = 0
- Highly multimodal (many local minima)
- Tests global search capability

Importing Functions
===================

All functions can be imported from the main module:

.. code-block:: python

    # Import specific functions
    from surfaces.test_functions import (
        SphereFunction,
        AckleyFunction,
        RosenbrockFunction,
    )

    # Import all algebraic functions
    from surfaces.test_functions.algebraic import algebraic_functions

    # List available functions
    for func_class in algebraic_functions:
        print(func_class.__name__)

Function Reference
==================

For complete API documentation of each function, see the
:doc:`API Reference </api_reference>`.
