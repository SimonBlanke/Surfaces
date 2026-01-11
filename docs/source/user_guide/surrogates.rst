.. _user_guide_surrogates:

================
Surrogate Models
================

Surfaces provides pre-trained surrogate models for expensive test functions.
Surrogates are fast approximations that can speed up optimization by
100-1000x while maintaining high accuracy.

Overview
========

Machine learning test functions require actual model training, making each
evaluation expensive (100ms to several seconds). Surrogate models are
pre-trained neural networks that approximate these expensive functions,
enabling rapid evaluation for algorithm development and benchmarking.

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - Metric
     - Without Surrogate
     - With Surrogate
   * - Evaluation time
     - 100-1000 ms
     - ~1 ms
   * - Typical speedup
     - ---
     - 100-1000x
   * - Accuracy (R²)
     - ---
     - 0.95+

When to Use Surrogates
======================

Surrogates are ideal for:

- **Algorithm development**: Test optimization algorithms rapidly
- **Benchmarking**: Run many trials without waiting for real training
- **Visualization**: Generate loss landscapes quickly
- **Parameter studies**: Explore optimizer hyperparameters

Surrogates are NOT recommended when:

- You need exact evaluation values (not approximations)
- You're validating final optimization results
- The function is already fast (algebraic functions)

Basic Usage
===========

Enable surrogates by setting ``use_surrogate=True`` when creating a function:

.. code-block:: python

    from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

    # Real evaluation (~200ms per call)
    func_real = KNeighborsClassifierFunction(dataset="iris", cv=5)

    # Surrogate evaluation (~1ms per call)
    func_fast = KNeighborsClassifierFunction(dataset="iris", cv=5, use_surrogate=True)

    # Both have the same interface
    params = {"n_neighbors": 5, "algorithm": "auto"}

    score_real = func_real(params)   # Slow but exact
    score_fast = func_fast(params)   # Fast approximation

The surrogate returns an approximation of what the real function would return.
For well-trained surrogates, the difference is typically less than 0.02 in
absolute score.

Fallback Behavior
-----------------

If no surrogate model is available for a function, the function automatically
falls back to real evaluation with a warning:

.. code-block:: python

    # If no surrogate exists for SVMClassifierFunction
    func = SVMClassifierFunction(use_surrogate=True)
    # UserWarning: No surrogate model found, falling back to real evaluation

Surrogate Coverage
==================

Not all ML functions have pre-trained surrogates yet. The following table
shows which functions have surrogates available, along with their accuracy
and speedup metrics.

.. include:: /_generated/surrogates/coverage.rst

How Surrogates Work
===================

Technical Details
-----------------

Surrogates are trained by:

1. **Data collection**: Evaluating the real function across a grid of
   hyperparameter and fixed-parameter combinations
2. **Model training**: Fitting a neural network (MLP) to approximate
   the hyperparameter-to-score mapping
3. **Export**: Saving the model in ONNX format for fast inference

The surrogate includes:

- **ONNX model**: The trained MLP regressor
- **Metadata**: Parameter encodings, training statistics
- **Validity model** (optional): Classifier to detect invalid parameter combinations

Fixed Parameters
----------------

ML functions have two types of parameters:

- **Hyperparameters**: Search space parameters (e.g., ``n_neighbors``, ``algorithm``)
- **Fixed parameters**: Set at initialization (e.g., ``dataset``, ``cv``)

Surrogates are trained across all combinations of fixed parameters, so a single
surrogate file covers all datasets and CV folds:

.. code-block:: python

    # Same surrogate works for different datasets
    func1 = KNeighborsClassifierFunction(dataset="iris", use_surrogate=True)
    func2 = KNeighborsClassifierFunction(dataset="digits", use_surrogate=True)
    func3 = KNeighborsClassifierFunction(dataset="wine", cv=10, use_surrogate=True)

Validating Accuracy
===================

You can validate surrogate accuracy against real function evaluations:

.. code-block:: python

    from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction
    from surfaces._surrogates import SurrogateValidator

    # Create real function (not surrogate)
    func = KNeighborsClassifierFunction(dataset="iris", cv=5)

    # Create validator
    validator = SurrogateValidator(func)

    # Run validation with random samples
    results = validator.validate_random(n_samples=100)

    # Access metrics
    print(f"R² Score: {results['metrics']['r2']:.4f}")
    print(f"Speedup:  {results['timing']['speedup']:.0f}x")

Validation output includes:

- **Accuracy metrics**: R², MAE, RMSE, max error, correlation
- **Timing**: Average real/surrogate time, speedup factor
- **Raw data**: Arrays of real vs predicted values for plotting

Installation
============

Surrogates require the ``surrogates`` extra and ONNX runtime:

.. code-block:: bash

    pip install surfaces[surrogates]

This installs:

- ``surfaces-onnx-files``: Pre-trained model files
- ``onnxruntime``: Fast ONNX model inference

Performance Tips
================

1. **Reuse function instances**: Creating a surrogate function loads the ONNX
   model once; reuse the instance for many evaluations.

2. **Batch evaluation**: Use ``.batch()`` for evaluating many points at once:

   .. code-block:: python

       params_list = [{"n_neighbors": k} for k in range(1, 21)]
       results = func.batch(params_list)

3. **Check availability first**: Avoid the fallback warning by checking:

   .. code-block:: python

       from surfaces._surrogates import get_surrogate_path

       if get_surrogate_path("k_neighbors_classifier"):
           # Surrogate exists, safe to use
           func = KNeighborsClassifierFunction(use_surrogate=True)

Developer API
=============

For training new surrogates, see the developer documentation. The training
API includes:

.. code-block:: python

    from surfaces._surrogates import (
        train_ml_surrogate,       # Train one surrogate
        train_all_ml_surrogates,  # Train all registered
        train_missing_ml_surrogates,  # Train only missing
        list_ml_surrogates,       # Check status
    )

    # Train a specific surrogate
    path = train_ml_surrogate("k_neighbors_classifier", verbose=True)

    # Check which surrogates exist
    status = list_ml_surrogates()
    for name, info in status.items():
        print(f"{name}: {'available' if info['exists'] else 'missing'}")
