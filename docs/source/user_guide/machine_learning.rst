.. _user_guide_machine_learning:

==========================
Machine Learning Functions
==========================

Surfaces provides test functions based on real machine learning models.
These functions offer realistic hyperparameter optimization landscapes
derived from actual model training tasks.

Overview
========

ML-based test functions evaluate the performance of machine learning
models with given hyperparameters. Unlike mathematical functions with
known global optima, these functions represent realistic optimization
challenges.

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Function
     - Task
     - Model
   * - KNeighborsClassifierFunction
     - Classification
     - K-Nearest Neighbors
   * - KNeighborsRegressorFunction
     - Regression
     - K-Nearest Neighbors
   * - GradientBoostingRegressorFunction
     - Regression
     - Gradient Boosting

Why ML-Based Functions?
=======================

Traditional mathematical test functions are useful but don't capture
the characteristics of real hyperparameter optimization:

- **Noise**: Real ML evaluations have variance from data splits
- **Discrete parameters**: Many hyperparameters are categorical
- **Complex interactions**: Parameters often interact non-linearly
- **Expensive**: Real training takes significant time

Surfaces' ML functions provide these realistic properties in a
standardized, reproducible way.

Classification Functions
========================

KNeighborsClassifierFunction
----------------------------

Optimizes K-Nearest Neighbors classification hyperparameters:

.. code-block:: python

    from surfaces.test_functions import KNeighborsClassifierFunction

    # Create the function
    func = KNeighborsClassifierFunction()

    # Evaluate with hyperparameters
    score = func({
        "n_neighbors": 5,
        "weights": "distance",
        "p": 2
    })

    print(f"Accuracy: {score}")

**Hyperparameters**:

- ``n_neighbors``: Number of neighbors (integer)
- ``weights``: Weight function ('uniform' or 'distance')
- ``p``: Power parameter for Minkowski distance (1 = Manhattan, 2 = Euclidean)

**Returns**: Classification accuracy (higher is better when using score mode)

Regression Functions
====================

KNeighborsRegressorFunction
---------------------------

Optimizes K-Nearest Neighbors regression hyperparameters:

.. code-block:: python

    from surfaces.test_functions import KNeighborsRegressorFunction

    func = KNeighborsRegressorFunction()

    score = func({
        "n_neighbors": 7,
        "weights": "uniform",
        "p": 1
    })

**Hyperparameters**:

- ``n_neighbors``: Number of neighbors
- ``weights``: Weight function
- ``p``: Minkowski distance power

**Returns**: R² score or negative MSE depending on mode

GradientBoostingRegressorFunction
---------------------------------

Optimizes Gradient Boosting regression hyperparameters:

.. code-block:: python

    from surfaces.test_functions import GradientBoostingRegressorFunction

    func = GradientBoostingRegressorFunction()

    score = func({
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1
    })

**Hyperparameters**:

- ``n_estimators``: Number of boosting stages
- ``max_depth``: Maximum tree depth
- ``learning_rate``: Shrinkage parameter

**Returns**: R² score or negative MSE depending on mode

Using ML Functions
==================

Loss vs Score Mode
------------------

ML functions naturally return scores (higher is better). The metric
parameter controls the sign:

.. code-block:: python

    func = KNeighborsClassifierFunction(metric="score")
    score = func(params)  # Returns accuracy (0-1)

    func = KNeighborsClassifierFunction(metric="loss")
    loss = func(params)  # Returns negative accuracy

Getting the Search Space
------------------------

The search space includes both continuous and categorical parameters:

.. code-block:: python

    func = KNeighborsClassifierFunction()
    space = func.search_space()

    print(space.keys())
    # dict_keys(['n_neighbors', 'weights', 'p'])

    print(space['weights'])
    # ['uniform', 'distance']

scipy Integration Limitations
-----------------------------

ML functions with categorical parameters cannot be directly converted
to scipy format. Use optimization libraries that support mixed
parameter types:

.. code-block:: python

    # This works for continuous-only functions
    # func.to_scipy()

    # For ML functions, use libraries like:
    # - Hyperactive
    # - Optuna
    # - scikit-optimize

Performance Considerations
==========================

ML function evaluations involve actual model training, so they're
slower than mathematical functions:

- **KNeighbors**: Fast (milliseconds per evaluation)
- **GradientBoosting**: Slower (seconds per evaluation)

For benchmarking optimization algorithms, consider:

1. Using fewer iterations
2. Testing on faster functions (KNeighbors)
3. Adding artificial delays to mathematical functions for comparison

Importing Functions
===================

.. code-block:: python

    # Import specific functions
    from surfaces.test_functions import (
        KNeighborsClassifierFunction,
        KNeighborsRegressorFunction,
        GradientBoostingRegressorFunction,
    )

    # Import all ML functions
    from surfaces.test_functions.machine_learning import machine_learning_functions

    for func_class in machine_learning_functions:
        print(func_class.__name__)
