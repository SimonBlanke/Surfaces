.. _user_guide_bbob:

==============
BBOB Functions
==============

The Black-Box Optimization Benchmarking (BBOB) suite is part of the
COCO (Comparing Continuous Optimizers) platform. It is the standard
benchmark for comparing continuous optimization algorithms.

----

What is BBOB?
=============

BBOB is a carefully designed benchmark suite with:

- **24 noiseless functions**: Covering different optimization challenges
- **Systematic design**: Functions grouped by difficulty characteristics
- **Standard methodology**: Results comparable across publications
- **Competition use**: Used in GECCO BBOB workshops

When you benchmark on BBOB, you can compare your results directly
with hundreds of published optimizers.

----

Function Groups
===============

BBOB functions are organized into five groups based on their characteristics:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Group
     - Characteristics
     - Example Functions
   * - **Separable**
     - Variables independent
     - Sphere, Ellipsoidal
   * - **Low/Moderate Conditioning**
     - Manageable condition numbers
     - Rosenbrock, Attractive Sector
   * - **High Conditioning**
     - Ill-conditioned
     - Bent Cigar, Sharp Ridge
   * - **Multi-modal (adequate)**
     - Multiple optima, exploitable structure
     - Rastrigin, Schaffer F7
   * - **Multi-modal (weak)**
     - Many optima, weak global structure
     - Schwefel, Gallagher

----

Available BBOB Functions
========================

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Function
     - Group
     - Properties
   * - ``Sphere``
     - Separable
     - Simplest, baseline
   * - ``EllipsoidalSeparable``
     - Separable
     - Ill-conditioned, separable
   * - ``RastriginSeparable``
     - Multi-modal
     - Highly multimodal, separable
   * - ``BuecheRastrigin``
     - Multi-modal
     - Asymmetric Rastrigin
   * - ``LinearSlope``
     - Separable
     - Linear function
   * - ``AttractiveSector``
     - Low conditioning
     - Sector-based optima
   * - ``StepEllipsoidal``
     - Low conditioning
     - Discrete steps
   * - ``RosenbrockOriginal``
     - Low conditioning
     - Classic Rosenbrock
   * - ``RosenbrockRotated``
     - Low conditioning
     - Rotated variant
   * - ``EllipsoidalRotated``
     - High conditioning
     - Rotated, ill-conditioned
   * - ``Discus``
     - High conditioning
     - Single sensitive direction
   * - ``BentCigar``
     - High conditioning
     - Narrow ridge
   * - ``SharpRidge``
     - High conditioning
     - Sharp narrow ridge
   * - ``DifferentPowers``
     - High conditioning
     - Variable sensitivities
   * - ``RastriginRotated``
     - Multi-modal
     - Rotated Rastrigin
   * - ``Weierstrass``
     - Multi-modal
     - Rugged landscape
   * - ``SchaffersF7``
     - Multi-modal
     - Asymmetric, ill-conditioned
   * - ``SchaffersF7Ill``
     - Multi-modal
     - More ill-conditioned
   * - ``GriewankRosenbrock``
     - Multi-modal
     - Combined landscape
   * - ``Schwefel``
     - Multi-modal (weak)
     - Deceptive optima
   * - ``Gallagher101``
     - Multi-modal (weak)
     - 101 random peaks
   * - ``Gallagher21``
     - Multi-modal (weak)
     - 21 random peaks
   * - ``Katsuura``
     - Multi-modal (weak)
     - Fractal-like
   * - ``LunacekBiRastrigin``
     - Multi-modal (weak)
     - Two funnels

----

Usage Example
=============

.. code-block:: python

    from surfaces.test_functions.bbob import (
        Sphere,
        RosenbrockRotated,
        RastriginRotated,
        Schwefel,
    )

    # Create functions
    sphere = Sphere(n_dim=10)
    rosenbrock = RosenbrockRotated(n_dim=10)
    rastrigin = RastriginRotated(n_dim=10)
    schwefel = Schwefel(n_dim=10)

    # Evaluate
    for func in [sphere, rosenbrock, rastrigin, schwefel]:
        result = func(func.search_space_sample())
        print(f"{func.__class__.__name__}: {result:.4f}")

----

Why Use BBOB?
=============

Comparable Results
------------------

BBOB has been used in optimization research for over a decade. When you
report results on BBOB functions, readers can directly compare with:

- Published papers
- COCO data archive
- Your own previous experiments

Systematic Coverage
-------------------

BBOB functions systematically cover different optimization challenges:

- Separability
- Conditioning
- Multi-modality
- Global structure

This ensures you test all aspects of your optimizer.

Standard Methodology
--------------------

The COCO platform provides standardized:

- Performance measures (ERT, success rate)
- Visualization tools
- Statistical comparisons

----

BBOB vs Algebraic
=================

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Algebraic
     - BBOB
   * - Purpose
     - General benchmarking
     - Rigorous comparison
   * - Standardization
     - Varies by paper
     - COCO standard
   * - Transformations
     - Usually none
     - Shifted, rotated
   * - Comparability
     - Limited
     - Extensive literature

----

Next Steps
==========

- :doc:`cec` - CEC competition benchmarks
- :doc:`/api_reference/test_functions/bbob` - Complete API reference
- `COCO Platform <https://github.com/numbbo/coco>`_ - Official BBOB implementation
