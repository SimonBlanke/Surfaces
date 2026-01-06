.. _user_guide_test_functions:

==============
Test Functions
==============

Surfaces provides five categories of test functions, each designed for
specific benchmarking purposes.

----

Function Categories
===================

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item-card:: Algebraic Functions
      :link: algebraic/index
      :link-type: doc
      :class-card: sd-border-primary

      Classic mathematical test functions from the optimization literature.
      Well-studied, with known properties and analytical formulas.

      - **1D Functions**: Simple univariate problems
      - **2D Functions**: Visualizable landscapes
      - **N-D Functions**: Scalable to any dimension

   .. grid-item-card:: BBOB Functions
      :link: bbob
      :link-type: doc
      :class-card: sd-border-success

      Black-Box Optimization Benchmarking suite from the COCO platform.
      The standard for rigorous optimizer comparison.

      - 24 noiseless functions
      - Used in GECCO competitions
      - Systematic difficulty progression

   .. grid-item-card:: CEC Functions
      :link: cec
      :link-type: doc
      :class-card: sd-border-warning

      Competition on Evolutionary Computation benchmark suites.
      Challenging functions for advanced optimizer testing.

      - CEC 2013, 2014, 2017 suites
      - Shifted and rotated variants
      - Composition functions

   .. grid-item-card:: Machine Learning Functions
      :link: machine_learning
      :link-type: doc
      :class-card: sd-border-info

      Test functions based on real ML model training.
      Realistic hyperparameter optimization landscapes.

      - Classification and regression
      - Tabular, image, time series
      - Surrogate models available

   .. grid-item-card:: Engineering Functions
      :link: engineering
      :link-type: doc
      :class-card: sd-border-danger

      Real-world constrained engineering design problems.
      Physical meaning and practical relevance.

      - Welded Beam Design
      - Pressure Vessel Design
      - Tension-Compression Spring

----

Choosing the Right Category
===========================

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Use Case
     - Recommended Category
     - Why
   * - Quick prototyping
     - Algebraic (2D)
     - Fast, visualizable, simple
   * - Rigorous comparison
     - BBOB
     - Standard benchmark, comparable results
   * - Algorithm competition
     - CEC
     - Used in IEEE CEC competitions
   * - Real-world relevance
     - Machine Learning
     - Actual HPO landscapes
   * - Constrained optimization
     - Engineering
     - Physical constraints

----

Common Interface
================

All test functions share the same interface:

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction, KNeighborsClassifierFunction

    # Same interface for algebraic and ML functions
    for func_class in [SphereFunction, KNeighborsClassifierFunction]:
        func = func_class() if func_class == KNeighborsClassifierFunction else func_class(n_dim=3)

        # Evaluate
        result = func(func.search_space_sample())

        # Get search space
        space = func.search_space()

        # Convert to scipy
        objective, bounds, x0 = func.to_scipy()

----

.. toctree::
   :maxdepth: 2
   :hidden:

   algebraic/index
   bbob
   cec
   machine_learning
   engineering
