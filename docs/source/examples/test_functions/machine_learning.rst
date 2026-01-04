.. _example_machine_learning:

==========================
Machine Learning Functions
==========================

Examples using ML-based test functions.

.. contents:: On this page
   :local:
   :depth: 2

----

Basic ML Function
=================

.. code-block:: python

    from surfaces.test_functions import KNeighborsClassifierFunction

    # Create ML-based test function
    func = KNeighborsClassifierFunction()

    # Evaluate with hyperparameters
    params = {
        "n_neighbors": 5,
        "weights": "distance",
        "p": 2
    }
    score = func(params)
    print(f"Accuracy: {score:.4f}")

----

Search Space with Categoricals
==============================

.. code-block:: python

    func = KNeighborsClassifierFunction()
    space = func.search_space()

    print("Search space:")
    for name, values in space.items():
        if hasattr(values, 'min'):
            print(f"  {name}: [{values.min()}, {values.max()}] (numeric)")
        else:
            print(f"  {name}: {values} (categorical)")

----

Multiple ML Functions
=====================

.. code-block:: python

    from surfaces.test_functions import (
        KNeighborsClassifierFunction,
        KNeighborsRegressorFunction,
        GradientBoostingRegressorFunction,
    )

    functions = [
        KNeighborsClassifierFunction(),
        KNeighborsRegressorFunction(),
        GradientBoostingRegressorFunction(),
    ]

    for func in functions:
        sample = func.search_space_sample()
        result = func(sample)
        print(f"{func.__class__.__name__}: {result:.4f}")

----

.. note::

    ML functions involve actual model training and are slower than
    algebraic functions. For fast benchmarking, consider using
    surrogate models.
