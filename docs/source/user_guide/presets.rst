.. _user_guide_collection:

==========
Collection
==========

The collection module provides a unified interface to browse, filter, and
select test functions. It includes pre-defined suites for common benchmarking
scenarios.

----

Why Use Collection?
===================

Instead of manually selecting functions:

.. code-block:: python

    # Manual selection - tedious
    from surfaces.test_functions.algebraic import (
        SphereFunction,
        RastriginFunction,
        RosenbrockFunction,
        AckleyFunction,
        # ... many more
    )

    functions = [
        SphereFunction(n_dim=10),
        RastriginFunction(n_dim=10),
        # ...
    ]

Use the collection:

.. code-block:: python

    # With collection - simple
    from surfaces import collection

    functions = collection.standard.instantiate(n_dim=10)

----

Pre-defined Suites
==================

Quick Suite
-----------

Fast sanity check (5 functions) for smoke tests and development iteration.

.. code-block:: python

    from surfaces import collection

    functions = collection.quick.instantiate(n_dim=10)

    for func in functions:
        print(f"{func.__class__.__name__}")

**Includes:** Sphere, Ackley, Rosenbrock, Rastrigin, Griewank

Standard Suite
--------------

Academic comparison (15 functions) covering diverse landscape types.

.. code-block:: python

    from surfaces import collection

    functions = collection.standard.instantiate(n_dim=10)

**Includes:** Classic functions for publication-quality benchmarking.

BBOB Suite
----------

Full COCO/BBOB benchmark (24 functions) for standardized comparison.

.. code-block:: python

    from surfaces import collection

    functions = collection.bbob.instantiate(n_dim=10)

CEC Suites
----------

IEEE CEC competition benchmarks.

.. code-block:: python

    from surfaces import collection

    # CEC 2014 (30 functions)
    functions_2014 = collection.cec2014.instantiate(n_dim=10)

    # CEC 2017 (10 functions)
    functions_2017 = collection.cec2017.instantiate(n_dim=10)

Engineering Suite
-----------------

Constrained engineering design problems (5 functions).

.. code-block:: python

    from surfaces import collection

    functions = collection.engineering.instantiate()

    for func in functions:
        print(f"{func.__class__.__name__}: {len(func.constraints)} constraints")

----

Filtering Functions
===================

Filter functions by properties:

.. code-block:: python

    from surfaces import collection

    # All unimodal functions
    unimodal = collection.filter(unimodal=True)

    # Convex algebraic functions
    convex = collection.filter(category="algebraic", convex=True)

    # Separable functions
    separable = collection.filter(separable=True)

Available filter criteria:

- ``category``: "algebraic", "bbob", "cec", "engineering", "ml"
- ``unimodal``: True/False
- ``convex``: True/False
- ``separable``: True/False
- ``scalable``: True/False
- ``n_dim``: Specific dimension or None for variable

----

Searching Functions
===================

Search by name or tagline:

.. code-block:: python

    from surfaces import collection

    # Find all Rastrigin variants
    rastrigin = collection.search("rastrigin")
    print(f"Found {len(rastrigin)} Rastrigin functions")

    for func_cls in rastrigin:
        print(f"  - {func_cls.__name__}")

----

Set Operations
==============

Combine collections using set operations:

.. code-block:: python

    from surfaces import collection

    # Union: all functions from both suites
    combined = collection.quick + collection.engineering

    # Intersection: functions in both
    common = collection.filter(scalable=True) & collection.bbob

    # Difference: remove engineering from all
    non_engineering = collection - collection.engineering

----

Using with Optimizers
=====================

.. code-block:: python

    from surfaces import collection
    from gradient_free_optimizers import BayesianOptimizer

    functions = collection.standard.instantiate(n_dim=10)
    results = {}

    for func in functions:
        opt = BayesianOptimizer(func.search_space)
        opt.search(func, n_iter=100)

        results[func.__class__.__name__] = opt.best_score

    # Print results table
    for name, score in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:25s}: {score:.6f}")

----

Listing Available Suites
========================

.. code-block:: python

    from surfaces import collection

    for name, count in collection.list_suites().items():
        print(f"{name}: {count} functions")

----

Iterating Over All Functions
============================

The collection itself contains all available functions:

.. code-block:: python

    from surfaces import collection

    print(f"Total functions: {len(collection)}")

    # Iterate over all
    for func_cls in collection:
        print(func_cls.__name__)

----

Next Steps
==========

- :doc:`test_functions/index` - All available test functions
- :doc:`integrations/index` - Use collection with optimization frameworks
- :doc:`/api_reference/collection` - Complete collection API
