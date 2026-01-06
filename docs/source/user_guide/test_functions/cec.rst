.. _user_guide_cec:

=============
CEC Functions
=============

The CEC (Competition on Evolutionary Computation) benchmark suites are
challenging test functions used in IEEE CEC competitions. They are
designed to test advanced optimization algorithms.

----

What is CEC?
============

CEC benchmarks are:

- **Competition-grade**: Used in IEEE CEC annual competitions
- **Challenging**: Shifted optima, rotations, compositions
- **Standardized**: Specific rules for fair comparison
- **Evolving**: New suites released periodically (2013, 2014, 2017, ...)

If your optimizer performs well on CEC functions, it can compete
with state-of-the-art algorithms.

----

Available CEC Suites
====================

CEC 2013
--------

28 benchmark functions for real-parameter optimization.

**Basic Functions (f1-f5):**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - ``Rastrigin``
     - Separable, multimodal
   * - ``RotatedRastrigin``
     - Non-separable variant
   * - ``RotatedDiscus``
     - Ill-conditioned
   * - ``RotatedBentCigar``
     - Ridge structure
   * - ``DifferentPowers``
     - Variable sensitivities

**Multimodal Functions (f6-f20):**

- Rotated versions of classic functions
- Shifted global optima
- Non-separable landscapes

**Composition Functions (f21-f28):**

- ``CompositionFunction1`` through ``CompositionFunction8``
- Combine multiple basic functions
- Create complex multi-basin landscapes

CEC 2014
--------

30 benchmark functions with improved characteristics:

- Better coverage of difficulty features
- Scalable to different dimensions
- Standardized bounds and optima

CEC 2017
--------

Updated suite with modern test functions:

- Hybrid functions
- Updated composition rules
- New challenging landscapes

----

Key Features
============

Shifted Optima
--------------

Unlike basic algebraic functions where the optimum is at the origin,
CEC functions have shifted optima:

.. code-block:: python

    from surfaces.test_functions.benchmark.cec import RotatedRastrigin

    func = RotatedRastrigin(n_dim=10)

    # Optimum is NOT at origin
    result_origin = func({f"x{i}": 0.0 for i in range(10)})
    print(f"f(0, ..., 0) = {result_origin}")  # Not the minimum!

This tests whether your optimizer searches the entire space,
not just around the origin.

Rotated Search Spaces
---------------------

Many CEC functions apply rotation matrices:

- Variables become correlated
- Separable algorithms fail
- Tests true multi-dimensional search

Composition Functions
---------------------

Composition functions combine multiple basic functions:

.. code-block:: python

    from surfaces.test_functions.benchmark.cec import CompositionFunction1

    func = CompositionFunction1(n_dim=10)

    # Complex landscape with multiple basins
    result = func(func.search_space_sample())

These create realistic multi-modal landscapes where different
regions have different characteristics.

----

Usage Example
=============

.. code-block:: python

    from surfaces.test_functions.benchmark.cec.cec2013 import (
        RotatedRastrigin,
        RotatedBentCigar,
        CompositionFunction1,
    )

    # Standard 10D benchmark
    functions = [
        RotatedRastrigin(n_dim=10),
        RotatedBentCigar(n_dim=10),
        CompositionFunction1(n_dim=10),
    ]

    for func in functions:
        # Run your optimizer
        result = func(func.search_space_sample())
        print(f"{func.__class__.__name__}: {result:.4f}")

----

CEC vs BBOB vs Algebraic
========================

.. list-table::
   :header-rows: 1
   :widths: 20 27 27 26

   * - Aspect
     - Algebraic
     - BBOB
     - CEC
   * - Difficulty
     - Easy-Medium
     - Medium-Hard
     - Hard
   * - Optima
     - Often at origin
     - Shifted
     - Shifted
   * - Rotation
     - Usually none
     - Some functions
     - Most functions
   * - Composition
     - No
     - Limited
     - Extensive
   * - Competition use
     - No
     - GECCO
     - IEEE CEC

----

Competition Preparation
=======================

If you're preparing for CEC competitions:

1. **Use the correct suite**: Check which year's functions are required
2. **Follow the rules**: Specific evaluation budgets, dimension settings
3. **Report correctly**: Use standard metrics (error values, success rate)

.. code-block:: python

    from surfaces.test_functions.benchmark.cec.cec2017 import (
        # Import the specific year's functions
        # ...
    )

    # Standard competition settings
    n_dim = 10  # or 30, 50, 100
    max_evaluations = 10000 * n_dim

----

Next Steps
==========

- :doc:`machine_learning` - ML-based test functions
- :doc:`/api_reference/test_functions/cec` - Complete API reference
- `IEEE CEC <https://www.ieee-cec.org/>`_ - Competition information
