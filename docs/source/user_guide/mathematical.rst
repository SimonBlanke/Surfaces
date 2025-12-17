.. _user_guide_mathematical:

======================
Mathematical Functions
======================

Surfaces provides classic mathematical optimization test functions from
the literature. These functions have well-known properties and global optima.

Overview
========

Mathematical functions are organized by dimensionality:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Category
     - Count
     - Description
   * - 1D Functions
     - 1
     - Single-variable functions
   * - 2D Functions
     - 18
     - Fixed two-dimensional functions
   * - N-D Functions
     - 5
     - Scalable to any dimension

1D Functions
============

These functions operate on a single variable.

Gramacy & Lee Function
----------------------

A simple 1D function with multiple local minima:

.. code-block:: python

    from surfaces.test_functions import GramacyAndLeeFunction

    func = GramacyAndLeeFunction()
    result = func({"x0": 0.5})

    # Get search space
    space = func.search_space()

.. _2d_functions:

2D Functions
============

Fixed two-dimensional functions for visualization and testing.

Ackley Function
---------------

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

Rosenbrock Function (2D)
------------------------

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

Beale Function
--------------

.. code-block:: python

    from surfaces.test_functions import BealeFunction

    func = BealeFunction()
    # Global minimum at (3, 0.5) with f(3, 0.5) = 0

Himmelblau's Function
---------------------

Has four identical local minima:

.. code-block:: python

    from surfaces.test_functions import HimmelblausFunction

    func = HimmelblausFunction()
    # Four global minima at approximately:
    # (3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)

Other 2D Functions
------------------

The following 2D functions are also available:

- ``BoothFunction`` - Global minimum at (1, 3)
- ``BukinFunctionN6`` - Global minimum at (-10, 1)
- ``CrossInTrayFunction`` - Four global minima
- ``DropWaveFunction`` - Global minimum at (0, 0)
- ``EasomFunction`` - Sharp global minimum at (pi, pi)
- ``EggholderFunction`` - Complex multimodal
- ``GoldsteinPriceFunction`` - Global minimum at (0, -1)
- ``HölderTableFunction`` - Four global minima
- ``LangermannFunction`` - Multiple local minima
- ``LeviFunctionN13`` - Global minimum at (1, 1)
- ``MatyasFunction`` - Global minimum at (0, 0)
- ``McCormickFunction`` - Global minimum at (-0.547, -1.547)
- ``SchafferFunctionN2`` - Global minimum at (0, 0)
- ``SimionescuFunction`` - Constrained optimization test
- ``ThreeHumpCamelFunction`` - Global minimum at (0, 0)

.. _nd_functions:

N-Dimensional Functions
=======================

These functions scale to any number of dimensions.

Sphere Function
---------------

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

Rastrigin Function
------------------

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

Rosenbrock Function (N-D)
-------------------------

Generalizes to any dimension:

.. code-block:: python

    from surfaces.test_functions import RosenbrockFunction

    func = RosenbrockFunction(n_dim=10)

    # Global minimum at (1, 1, ..., 1)
    params = {f"x{i}": 1.0 for i in range(10)}
    result = func(params)  # 0.0

Griewank Function
-----------------

Multimodal with component interactions:

.. code-block:: python

    from surfaces.test_functions import GriewankFunction

    func = GriewankFunction(n_dim=5)
    # Global minimum at origin

Styblinski-Tang Function
------------------------

.. code-block:: python

    from surfaces.test_functions import StyblinskiTangFunction

    func = StyblinskiTangFunction(n_dim=5)
    # Global minimum at (-2.903534, ..., -2.903534)

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

    # Import all mathematical functions
    from surfaces.test_functions.mathematical import mathematical_functions

    # List available functions
    for func_class in mathematical_functions:
        print(func_class.__name__)

Function Reference
==================

For complete API documentation of each function, see the
:doc:`API Reference </api_reference>`.
