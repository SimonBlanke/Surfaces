.. _curated_test_functions:

======================
Curated Test Functions
======================

Surfaces is not just a random collection of test functions. Every function
in this library has been carefully selected to provide meaningful benchmarks
for optimization algorithms.

----

Five Categories of Test Functions
=================================

Surfaces organizes test functions into five distinct categories, each serving
a specific purpose in optimization benchmarking.

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item-card:: Algebraic Functions
      :class-card: sd-border-primary

      Classic mathematical test functions from the optimization literature.
      These functions have well-known properties and are widely used in
      academic research.

      - **1D Functions**: Simple univariate problems
      - **2D Functions**: Visualizable landscapes (Ackley, Rastrigin, etc.)
      - **N-D Functions**: Scalable to any dimension

      +++
      :doc:`Learn more </user_guide/test_functions/algebraic>`

   .. grid-item-card:: BBOB Functions
      :class-card: sd-border-success

      The Black-Box Optimization Benchmarking (BBOB) suite from the COCO
      platform. An established standard for comparing continuous optimizers.

      - 24 noiseless functions
      - Designed for rigorous algorithm comparison
      - Used in GECCO competitions

      +++
      :doc:`Learn more </user_guide/test_functions/bbob>`

   .. grid-item-card:: CEC Functions
      :class-card: sd-border-warning

      Competition on Evolutionary Computation benchmark suites. These
      challenging functions are used in IEEE CEC competitions.

      - CEC 2013, 2014, 2017 suites
      - Shifted and rotated variants
      - Composition functions

      +++
      :doc:`Learn more </user_guide/test_functions/cec>`

   .. grid-item-card:: Machine Learning Functions
      :class-card: sd-border-info

      Test functions based on real ML model training. Benchmark your
      optimizer on actual hyperparameter optimization problems.

      - Classification and regression tasks
      - Tabular, image, and time series data
      - Realistic optimization landscapes

      +++
      :doc:`Learn more </user_guide/test_functions/machine_learning>`

   .. grid-item-card:: Engineering Functions
      :class-card: sd-border-danger

      Real-world constrained engineering design problems with physical
      meaning and practical relevance.

      - Welded Beam Design
      - Pressure Vessel Design
      - Tension-Compression Spring

      +++
      :doc:`Learn more </user_guide/test_functions/engineering>`

----

Why Curated Matters
===================

Standard Benchmarks
-------------------

CEC and BBOB are not arbitrary function collections. They are carefully
designed benchmark suites used by the research community to compare
optimization algorithms fairly.

When you use these functions, you can:

- Compare your results directly with published research
- Reproduce experiments from academic papers
- Trust that the functions test relevant algorithm properties

Real Problems
-------------

Algebraic functions like Sphere or Rastrigin are useful, but they do not
represent real optimization challenges. Surfaces includes:

- **ML hyperparameter tuning**: Actual model training as objective
- **Engineering design**: Physical constraints and multi-modal landscapes

These functions help you understand how your optimizer performs on
problems that matter.

----

Quick Example
=============

.. code-block:: python

    from surfaces.test_functions import (
        # Algebraic
        SphereFunction,
        RastriginFunction,
        # BBOB
        # RosenbrockRotated,
        # CEC
        # RotatedRastrigin,
        # Machine Learning
        KNeighborsClassifierFunction,
        # Engineering
        WeldedBeamDesign,
    )

    # All functions share the same interface
    for func_class in [SphereFunction, RastriginFunction]:
        func = func_class(n_dim=5)
        result = func(func.search_space_sample())
        print(f"{func.__class__.__name__}: {result:.4f}")

----

Next Steps
==========

- :doc:`/user_guide/test_functions/index` - Detailed guide to all function categories
- :doc:`/api_reference/test_functions/index` - Complete API reference
