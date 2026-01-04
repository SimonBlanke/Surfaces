.. _example_bbob:

==============
BBOB Functions
==============

Examples using BBOB benchmark functions.

.. contents:: On this page
   :local:
   :depth: 2

----

Basic BBOB Usage
================

.. code-block:: python

    from surfaces.test_functions.bbob import (
        Sphere,
        RosenbrockRotated,
        RastriginRotated,
        Schwefel,
    )

    # Create 10D functions
    functions = [
        Sphere(n_dim=10),
        RosenbrockRotated(n_dim=10),
        RastriginRotated(n_dim=10),
        Schwefel(n_dim=10),
    ]

    for func in functions:
        result = func(func.search_space_sample())
        print(f"{func.__class__.__name__}: {result:.4f}")

----

BBOB Function Groups
====================

.. code-block:: python

    """Evaluate functions from different BBOB groups."""

    from surfaces.test_functions.bbob import (
        # Separable
        Sphere,
        EllipsoidalSeparable,
        # Multimodal
        RastriginRotated,
        SchaffersF7,
        # Weak structure
        Schwefel,
        Gallagher101,
    )

    groups = {
        "Separable": [Sphere, EllipsoidalSeparable],
        "Multimodal": [RastriginRotated, SchaffersF7],
        "Weak structure": [Schwefel, Gallagher101],
    }

    for group_name, func_classes in groups.items():
        print(f"\n{group_name}:")
        for func_class in func_classes:
            func = func_class(n_dim=5)
            result = func(func.search_space_sample())
            print(f"  {func_class.__name__}: {result:.4f}")

----

.. note::

    BBOB functions are designed for rigorous algorithm comparison.
    See the COCO platform documentation for standard benchmarking protocols.
