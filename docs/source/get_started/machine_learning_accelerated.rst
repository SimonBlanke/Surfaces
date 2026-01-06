.. _machine_learning_accelerated:

============================
Machine Learning Accelerated
============================

Benchmarking optimizers on real machine learning problems is expensive.
Training a model takes time. Surfaces solves this with pre-trained
surrogate models that approximate expensive ML objective functions.

----

The Problem
===========

When you benchmark an optimizer on hyperparameter optimization:

.. code-block:: python

    # This is SLOW - trains a real model every evaluation
    def objective(params):
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)

Each evaluation requires:

- Loading data
- Training a model
- Validating performance
- Possibly repeating for cross-validation

A single evaluation can take seconds to minutes. Running 1000 optimizer
iterations becomes impractical.

----

The Solution: Surrogate Models
==============================

Surfaces provides pre-trained surrogate models that approximate the
objective function landscape. These surrogates are:

- **Fast**: Milliseconds per evaluation instead of seconds
- **Realistic**: Trained on actual hyperparameter search data
- **Deterministic**: Same input always gives same output (no training noise)

.. code-block:: python

    from surfaces.test_functions.machine_learning import KNeighborsClassifierSurrogate

    # This is FAST - uses pre-trained approximation
    func = KNeighborsClassifierSurrogate()
    result = func({"n_neighbors": 5, "weights": "distance", "p": 2})

----

How It Works
============

1. **Data Collection**: We run extensive hyperparameter searches on real datasets
2. **Model Training**: A surrogate model learns the mapping from hyperparameters to performance
3. **ONNX Export**: The surrogate is exported to ONNX format for fast inference
4. **Distribution**: The ONNX model is packaged with Surfaces

The surrogate captures the realistic landscape of ML hyperparameter
optimization without the computational cost.

----

Available Surrogates
====================

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Surrogate
     - Base Model
     - Use Case
   * - ``KNeighborsClassifierSurrogate``
     - K-Nearest Neighbors
     - Classification HPO
   * - ``GradientBoostingSurrogate``
     - Gradient Boosting
     - Regression HPO
   * - More coming soon...
     -
     -

----

Quick Example
=============

Compare optimizer performance on ML hyperparameter optimization:

.. code-block:: python

    from surfaces.test_functions.machine_learning import KNeighborsClassifierSurrogate
    from scipy.optimize import differential_evolution

    # Create the surrogate function
    func = KNeighborsClassifierSurrogate()

    # Get scipy-compatible format
    objective, bounds, _ = func.to_scipy()

    # Run 1000 evaluations in seconds, not hours
    result = differential_evolution(objective, bounds, maxiter=1000)
    print(f"Best hyperparameters found: {result.x}")

----

Installation
============

Surrogate models require the ONNX runtime:

.. code-block:: bash

    pip install surfaces[surrogates]

This installs:

- ``onnxruntime``: Fast inference engine
- ``surfaces-onnx``: Pre-trained surrogate model files

See :doc:`/installation/surrogates` for detailed installation instructions.

----

Next Steps
==========

- :doc:`/installation/surrogates` - Installing surrogate dependencies
- :doc:`/user_guide/test_functions/machine_learning` - ML function guide
- :doc:`/api_reference/test_functions/machine_learning` - API reference
