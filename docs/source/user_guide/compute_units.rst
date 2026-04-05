.. _user_guide_compute_units:

=============
Compute Units
=============

When benchmarking optimization algorithms, the number of iterations alone
does not tell the full story. A Bayesian optimizer that spends 5 seconds
fitting a Gaussian process per iteration looks wasteful on a function that
evaluates in microseconds, but reasonable on one that takes seconds. The
metric that matters is **total computational cost**, not iteration count.

Surfaces provides **Compute Units (CU)** as a hardware-independent cost
measure. Every test function carries a pre-computed ``eval_cost`` attribute,
and a calibration function lets you express any wall-clock measurement in
the same unit.


Why Not Wall-Clock Seconds?
===========================

Wall-clock time is machine-dependent. A benchmark run on a fast workstation
produces different numbers than the same run on a laptop, making results
hard to compare across publications or machines.

Compute Units solve this by normalizing all times against a **reference
operation** measured on the same machine. The reference combines arithmetic,
transcendental, and matrix-vector operations to represent a typical
computation mix. Since both the function evaluation and the reference
run on the same hardware, the ratio is approximately constant across
machines.


Accessing Evaluation Cost
=========================

Every test function has an ``eval_cost`` value in its spec:

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction

    func = SphereFunction(n_dim=2)
    print(func.spec.eval_cost)  # 0.1

This value represents the average cost of a single evaluation in CU,
measured with default parameters.

You can also access it through the spec dict:

.. code-block:: python

    func.spec.as_dict()["eval_cost"]  # 0.1


Typical Cost Ranges
===================

.. list-table::
   :widths: 35 25 40
   :header-rows: 1

   * - Category
     - CU Range
     - Examples
   * - Algebraic (standard)
     - 0.1 -- 1.4
     - Sphere (0.1), Rastrigin (0.2), Shekel (1.4)
   * - Engineering (constrained)
     - 0.3 -- 1.0
     - PressureVessel (0.3), WeldedBeam (1.0)
   * - BBOB
     - 0.5 -- 91.7
     - Sphere (0.5), Weierstrass (16.6), Katsuura (91.7)
   * - Simulation (ODE)
     - 400 -- 36,300
     - ConsecutiveReaction (433), RCFilter (36,300)
   * - ML (simple)
     - 500 -- 5,400
     - DecisionTreeClassifier (5,400), SVM (760)
   * - ML (complex)
     - 23,900 -- 2,428,100
     - RandomForest (72,100), GradientBoosting (2,428,100)

.. note::

   ML function costs were measured with default parameters (``dataset="digits"``,
   ``cv=5``). The actual cost depends heavily on the dataset size and cross-validation
   settings. A RandomForest on a large dataset can cost orders of magnitude more
   than the listed 72,100 CU. Treat these values as relative reference points for
   comparing ML functions against each other, not as predictions of wall-clock cost.


Filtering by Cost
=================

The collection system supports filtering by ``eval_cost``:

.. code-block:: python

    from surfaces import collection

    # All functions with eval_cost under 10 CU (fast functions)
    fast = collection.filter(eval_cost=lambda c: c is not None and c < 10)


Converting Optimizer Overhead
=============================

.. warning::

   The ``surfaces._cost`` module is intended for your own experiments.
   Its interface may change in future versions.

To compare function eval cost with optimizer overhead, use ``to_cu()``
to convert wall-clock seconds into the same unit:

.. code-block:: python

    import time
    from surfaces._cost import calibrate, to_cu

    # Calibrate once per session (~1 second)
    calibrate()

    # Measure optimizer overhead
    t0 = time.perf_counter()
    next_params = optimizer.ask()
    optimizer_seconds = time.perf_counter() - t0

    # Convert to CU
    optimizer_cu = to_cu(optimizer_seconds)

    # Now both are comparable
    print(f"Eval cost:      {func.spec.eval_cost:.1f} CU")
    print(f"Optimizer cost: {optimizer_cu:.1f} CU")


Benchmarking Example
====================

A complete benchmark loop tracking total compute in CU:

.. code-block:: python

    import time
    from surfaces._cost import calibrate, to_cu
    from surfaces.test_functions.algebraic import SphereFunction

    calibrate()
    func = SphereFunction(n_dim=5)

    total_eval_cu = 0.0
    total_optimizer_cu = 0.0
    history = []

    for i in range(budget):
        # Optimizer overhead
        t0 = time.perf_counter()
        params = optimizer.ask()
        total_optimizer_cu += to_cu(time.perf_counter() - t0)

        # Function evaluation
        t0 = time.perf_counter()
        score = func(params)
        total_eval_cu += to_cu(time.perf_counter() - t0)

        optimizer.tell(params, score)

        total_cu = total_eval_cu + total_optimizer_cu
        history.append((total_cu, best_score))

    # Plot: Score vs Total Compute (CU)
    # This plot is hardware-independent and comparable across machines.

On a cheap function like Sphere (0.1 CU), an optimizer with high
per-iteration overhead (e.g. Bayesian optimization at ~500,000 CU)
spends 99.99% of the budget on itself. A simple hill climber at ~5 CU
overhead gets millions more evaluations in the same budget.

On an expensive function like GradientBoostingClassifier (~2,400,000 CU),
the optimizer overhead becomes negligible and the smarter algorithm wins.


The Reference Operation
=======================

The calibration function ``calibrate()`` measures a single reference
operation that combines three types of computation:

- **Arithmetic**: ``np.sum(x * x)`` on a 50-element vector
- **Transcendental**: ``np.sin(x)`` and ``np.exp(x)``
- **Matrix-vector product**: ``A @ x`` with a 50x50 matrix

One reference operation takes approximately 10 microseconds on modern
hardware. The ``calibrate()`` function runs adaptively: it measures
enough iterations to fill at least 1 second, ensuring sub-0.1%
measurement error regardless of CPU speed.

Results are cached for the session. Call ``reset()`` to force
re-calibration:

.. code-block:: python

    from surfaces._cost import calibrate, reset

    ref_time = calibrate()       # ~8e-6 seconds on a modern CPU
    reset()                      # clear cache
    ref_time = calibrate()       # re-measure


Updating eval_cost Values
=========================

When adding new test functions or re-calibrating existing ones, use the
provided script:

.. code-block:: bash

    # Measure all functions and write values to source files
    python scripts/calibrate_eval_costs.py

    # Dry run (measure only, no file changes)
    python scripts/calibrate_eval_costs.py --dry-run

    # Adjust measurement duration and timeout
    python scripts/calibrate_eval_costs.py --min-duration 1.0 --timeout 60

The script auto-discovers all available test functions, measures each one
adaptively, and updates the ``eval_cost`` value in each class's ``_spec``
dict. Functions that require unavailable dependencies (e.g. tensorflow,
surfaces-cec-data) are skipped with a clear message.
