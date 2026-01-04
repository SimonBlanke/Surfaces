.. _installation_cec:

=============
CEC Functions
=============

The CEC (Competition on Evolutionary Computation) benchmark suites are
included in the core installation. This page covers additional setup
that may be required for certain CEC functions.

----

Installation
============

CEC functions are included in the core installation:

.. code-block:: bash

    pip install surfaces

No additional dependencies required.

----

Available CEC Suites
====================

Surfaces includes functions from multiple CEC competitions:

CEC 2013
--------

28 benchmark functions for real-parameter optimization.

.. code-block:: python

    from surfaces.test_functions.cec import (
        # Basic functions
        Rastrigin,
        RotatedRastrigin,
        RotatedDiscus,
        RotatedBentCigar,
        # Composition functions
        CompositionFunction1,
        CompositionFunction2,
        # ... and more
    )

CEC 2014
--------

30 benchmark functions with improved characteristics.

.. code-block:: python

    from surfaces.test_functions.cec import (
        # CEC 2014 functions
        # ...
    )

CEC 2017
--------

Updated benchmark suite with modern test functions.

.. code-block:: python

    from surfaces.test_functions.cec import (
        # CEC 2017 functions
        # ...
    )

----

Usage Example
=============

.. code-block:: python

    from surfaces.test_functions.cec import RotatedRastrigin

    # Create a 10-dimensional function
    func = RotatedRastrigin(n_dim=10)

    # Evaluate
    result = func(func.search_space_sample())
    print(f"Result: {result}")

    # Get search space
    space = func.search_space()

----

What CEC Functions Offer
========================

Shifted Optima
--------------

Unlike basic algebraic functions where the optimum is often at the origin,
CEC functions have shifted optima at unknown locations. This tests whether
your optimizer can find solutions anywhere in the search space.

Rotated Search Spaces
---------------------

Many CEC functions apply rotation matrices to make the landscape
non-separable. This tests whether your optimizer can handle
correlated variables.

Composition Functions
---------------------

CEC includes composition functions that combine multiple basic functions.
These create complex multi-modal landscapes that are particularly
challenging for optimization algorithms.

----

Next Steps
==========

- :doc:`/user_guide/test_functions/cec` - Detailed guide to CEC functions
- :doc:`/api_reference/test_functions/cec` - API reference
