.. _user_guide_algebraic:

===================
Algebraic Functions
===================

Algebraic functions are classic mathematical test functions from the
optimization literature. They have analytical formulas, known global
optima, and well-studied properties.

----

Why Algebraic Functions?
========================

These functions are the foundation of optimization benchmarking:

- **Well-understood**: Decades of research on their properties
- **Fast evaluation**: Simple formulas, microsecond evaluation
- **Visualizable**: 2D functions can be plotted as surfaces
- **Scalable**: Many extend to arbitrary dimensions

----

Function Dimensions
===================

.. grid:: 1 1 3 3
   :gutter: 4

   .. grid-item-card:: 1D Functions
      :link: 1d
      :link-type: doc

      Univariate test functions for simple benchmarks
      and visualization of optimizer behavior.

      - Gramacy & Lee
      - Forrester
      - And more...

   .. grid-item-card:: 2D Functions
      :link: 2d
      :link-type: doc

      Two-dimensional functions that can be visualized
      as 3D surfaces and contour plots.

      - Ackley, Rastrigin
      - Himmelblau, Rosenbrock
      - And more...

   .. grid-item-card:: N-D Functions
      :link: nd
      :link-type: doc

      Scalable functions that work in any dimension.
      Test optimizer scaling behavior.

      - Sphere, Rastrigin
      - Rosenbrock, Griewank
      - And more...

----

Common Properties
=================

Unimodal vs Multimodal
----------------------

- **Unimodal**: Single global optimum (e.g., Sphere)
- **Multimodal**: Multiple local optima (e.g., Rastrigin, Ackley)

Multimodal functions test an optimizer's ability to escape local optima.

Separable vs Non-separable
--------------------------

- **Separable**: Variables can be optimized independently
- **Non-separable**: Variables interact (e.g., Rosenbrock)

Non-separable functions test an optimizer's ability to handle
variable interactions.

----

Quick Start
===========

.. code-block:: python

    from surfaces.test_functions.algebraic import (
        # 2D only
        AckleyFunction,
        HimmelblausFunction,
        # N-D (scalable)
        SphereFunction,
        RastriginFunction,
        RosenbrockFunction,
    )

    # 2D function
    ackley = AckleyFunction()
    result = ackley({"x0": 0.0, "x1": 0.0})

    # N-D function with custom dimensions
    sphere = SphereFunction(n_dim=10)
    result = sphere({f"x{i}": 0.0 for i in range(10)})

----

.. toctree::
   :maxdepth: 1
   :hidden:

   1d
   2d
   nd
