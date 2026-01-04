.. _installation_machine_learning:

================
Machine Learning
================

Machine Learning test functions are based on real ML model training.
They require additional dependencies beyond the core installation.

----

Installation
============

.. code-block:: bash

    pip install surfaces[ml]

This installs:

- **scikit-learn**: Core ML library for most functions
- **Additional ML libraries** as needed

----

What You Get
============

With the ML extra installed, you can use test functions based on
actual machine learning model training:

Tabular Data
------------

Classification and regression on tabular datasets.

.. code-block:: python

    from surfaces.test_functions import (
        KNeighborsClassifierFunction,
        KNeighborsRegressorFunction,
        GradientBoostingRegressorFunction,
    )

    # Hyperparameter optimization as test function
    func = KNeighborsClassifierFunction()

    # Evaluate with hyperparameters
    score = func({
        "n_neighbors": 5,
        "weights": "distance",
        "p": 2
    })

Image Data
----------

ML functions on image classification tasks.

.. code-block:: python

    from surfaces.test_functions import (
        # Image-based ML functions
        # ...
    )

Time Series
-----------

ML functions for time series forecasting.

.. code-block:: python

    from surfaces.test_functions import (
        # Time series ML functions
        # ...
    )

----

Why ML Functions?
=================

Realistic Landscapes
--------------------

Algebraic functions like Sphere or Rastrigin have smooth, well-behaved
landscapes. Real hyperparameter optimization is different:

- Discrete and continuous parameters mixed
- Noisy evaluations (different random seeds)
- Plateaus and sharp transitions
- Interactions between hyperparameters

ML functions in Surfaces provide these realistic characteristics.

Practical Relevance
-------------------

When you benchmark on ML functions, you test your optimizer on
problems that actually matter. If your optimizer works well on
``KNeighborsClassifierFunction``, it will likely work well on
real hyperparameter tuning tasks.

----

XGBoost Support
===============

For XGBoost-based test functions:

.. code-block:: bash

    pip install surfaces[ml] xgboost

.. code-block:: python

    from surfaces.test_functions import XGBoostClassifierFunction

    func = XGBoostClassifierFunction()
    score = func({
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    })

----

Usage Example
=============

.. code-block:: python

    from surfaces.test_functions import KNeighborsClassifierFunction
    from scipy.optimize import differential_evolution

    # Create ML-based test function
    func = KNeighborsClassifierFunction()

    # Convert to scipy format
    objective, bounds, x0 = func.to_scipy()

    # Optimize hyperparameters
    result = differential_evolution(objective, bounds, maxiter=50)
    print(f"Best hyperparameters: {result.x}")
    print(f"Best score: {-result.fun}")  # Negate because scipy minimizes

----

Performance Note
================

ML functions are slower than algebraic functions because they
actually train models. For fast benchmarking with realistic
landscapes, consider using :doc:`surrogates`.

----

Next Steps
==========

- :doc:`/user_guide/test_functions/machine_learning` - ML functions guide
- :doc:`surrogates` - Fast surrogate approximations
- :doc:`/api_reference/test_functions/machine_learning` - API reference
