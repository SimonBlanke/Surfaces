.. _installation_surrogates:

==========
Surrogates
==========

Surrogate models provide fast approximations of expensive test functions.
They are pre-trained neural networks exported to ONNX format.

----

Installation
============

.. code-block:: bash

    pip install surfaces[surrogates]

This installs:

- **onnxruntime**: Fast inference engine for ONNX models
- **surfaces-onnx**: Pre-trained surrogate model files

----

What You Get
============

Surrogate models approximate the objective function landscape of
expensive ML hyperparameter optimization problems:

.. code-block:: python

    from surfaces.test_functions.machine_learning import KNeighborsClassifierSurrogate

    # Fast evaluation - no actual model training
    func = KNeighborsClassifierSurrogate()

    # Milliseconds per evaluation
    result = func({
        "n_neighbors": 5,
        "weights": "distance",
        "p": 2
    })

----

Speed Comparison
================

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Function Type
     - Evaluation Time
     - Use Case
   * - Algebraic (Sphere, etc.)
     - ~0.01 ms
     - Basic benchmarking
   * - ML Function (actual training)
     - ~100-1000 ms
     - Realistic but slow
   * - Surrogate Model
     - ~1 ms
     - Realistic and fast

Surrogates give you the best of both worlds: realistic ML landscapes
with near-algebraic speed.

----

How Surrogates Work
===================

1. **Data Collection**: Extensive hyperparameter searches on real datasets
2. **Model Training**: A neural network learns the hyperparameter-to-score mapping
3. **ONNX Export**: The model is exported to ONNX for portable, fast inference
4. **Distribution**: Pre-trained models are packaged with ``surfaces-onnx``

The surrogate captures the realistic landscape characteristics:

- Multi-modality
- Hyperparameter interactions
- Discrete/continuous parameter mixing
- Plateaus and sharp transitions

----

Available Surrogates
====================

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Surrogate
     - Approximates
   * - ``KNeighborsClassifierSurrogate``
     - K-Nearest Neighbors hyperparameter tuning
   * - ``GradientBoostingSurrogate``
     - Gradient Boosting hyperparameter tuning
   * - More coming...
     - Additional ML models

----

Usage Example
=============

Benchmark an optimizer on realistic ML landscapes:

.. code-block:: python

    from surfaces.test_functions.machine_learning import KNeighborsClassifierSurrogate
    from gradient_free_optimizers import BayesianOptimization

    # Create surrogate function
    func = KNeighborsClassifierSurrogate()

    # Run 1000 evaluations in seconds
    opt = BayesianOptimization(func.search_space())
    opt.search(func, n_iter=1000)

    print(f"Best score: {opt.best_score}")
    print(f"Best params: {opt.best_para}")

----

When to Use Surrogates
======================

Use surrogates when you need:

- **Fast iteration**: Testing many optimizer configurations
- **Large-scale benchmarks**: Running thousands of evaluations
- **Reproducibility**: Deterministic evaluations (no training noise)
- **CI/CD integration**: Quick feedback in automated pipelines

Use actual ML functions when you need:

- **Ground truth**: Validating final results
- **New datasets**: Surrogates are trained on specific data

----

ONNX Runtime Notes
==================

CPU vs GPU
----------

By default, ``onnxruntime`` uses CPU inference, which is fast enough
for surrogate evaluation. GPU acceleration is available but typically
not needed:

.. code-block:: bash

    # CPU only (default, recommended)
    pip install surfaces[surrogates]

    # With GPU support (optional)
    pip install onnxruntime-gpu

Platform Support
----------------

ONNX Runtime supports:

- Linux (x86_64, aarch64)
- macOS (x86_64, arm64)
- Windows (x86_64)

----

Next Steps
==========

- :doc:`/get_started/machine_learning_accelerated` - Introduction to surrogates
- :doc:`/user_guide/test_functions/machine_learning` - ML functions guide
- :doc:`/api_reference/test_functions/machine_learning` - API reference
