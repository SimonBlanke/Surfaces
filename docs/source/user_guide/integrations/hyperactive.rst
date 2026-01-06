.. _user_guide_hyperactive:

==========
Hyperactive
==========

Hyperactive is an advanced optimization toolkit built on top of
Gradient-Free-Optimizers. It provides additional features for
hyperparameter optimization.

----

Installation
============

.. code-block:: bash

    pip install hyperactive

----

Basic Usage
===========

.. code-block:: python

    from hyperactive import Hyperactive
    from surfaces.test_functions.algebraic import RastriginFunction

    func = RastriginFunction(n_dim=5)

    hyper = Hyperactive()
    hyper.add_search(func, func.search_space(), n_iter=100)
    hyper.run()

    print(f"Best score: {hyper.best_score(func)}")
    print(f"Best params: {hyper.best_para(func)}")

----

Multiple Searches
=================

Run multiple optimizations in parallel:

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction, RastriginFunction, AckleyFunction

    sphere = SphereFunction(n_dim=5)
    rastrigin = RastriginFunction(n_dim=5)
    ackley = AckleyFunction()

    hyper = Hyperactive()

    # Add multiple searches
    hyper.add_search(sphere, sphere.search_space(), n_iter=100)
    hyper.add_search(rastrigin, rastrigin.search_space(), n_iter=100)
    hyper.add_search(ackley, ackley.search_space(), n_iter=100)

    # Run all in parallel
    hyper.run(max_workers=3)

    # Get results
    for func in [sphere, rastrigin, ackley]:
        print(f"{func.__class__.__name__}: {hyper.best_score(func):.6f}")

----

Optimizer Selection
===================

.. code-block:: python

    from hyperactive.optimizers import (
        RandomSearchOptimizer,
        BayesianOptimizer,
        ParticleSwarmOptimizer,
    )

    hyper = Hyperactive()

    # Different optimizers for different functions
    hyper.add_search(
        sphere,
        sphere.search_space(),
        optimizer=BayesianOptimizer(),
        n_iter=100,
    )

    hyper.add_search(
        rastrigin,
        rastrigin.search_space(),
        optimizer=ParticleSwarmOptimizer(population=20),
        n_iter=100,
    )

    hyper.run()

----

Memory and Warm Starting
========================

.. code-block:: python

    # Enable memory to track all evaluations
    hyper.add_search(
        func,
        func.search_space(),
        n_iter=100,
        memory=True,
    )
    hyper.run()

    # Access search history
    search_data = hyper.search_data(func)

    # Warm start from previous results
    hyper2 = Hyperactive()
    hyper2.add_search(
        func,
        func.search_space(),
        n_iter=100,
        memory_warm_start=search_data,
    )
    hyper2.run()

----

Progress Tracking
=================

.. code-block:: python

    hyper.add_search(
        func,
        func.search_space(),
        n_iter=100,
        verbosity=["progress_bar", "print_results"],
    )

----

Constraints
===========

Add constraints to the search:

.. code-block:: python

    def constraint(params):
        # Return True if valid
        return params["x0"] + params["x1"] < 5.0

    hyper.add_search(
        func,
        func.search_space(),
        n_iter=100,
        constraints=[constraint],
    )

----

Benchmarking Setup
==================

.. code-block:: python

    from hyperactive import Hyperactive
    from hyperactive.optimizers import BayesianOptimizer, RandomSearchOptimizer
    from surfaces.test_functions.algebraic import RastriginFunction
    import numpy as np

    func = RastriginFunction(n_dim=10)

    # Run multiple trials
    results = {"Bayesian": [], "Random": []}

    for seed in range(10):
        for name, opt in [("Bayesian", BayesianOptimizer()),
                          ("Random", RandomSearchOptimizer())]:
            hyper = Hyperactive()
            hyper.add_search(
                func,
                func.search_space(),
                optimizer=opt,
                n_iter=100,
                random_state=seed,
            )
            hyper.run()
            results[name].append(hyper.best_score(func))

    for name, scores in results.items():
        print(f"{name}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

----

Hyperactive vs GFO
==================

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - GFO
     - Hyperactive
   * - API
     - Low-level
     - High-level
   * - Parallel
     - Manual
     - Built-in
   * - Warm start
     - Manual
     - Built-in
   * - Constraints
     - No
     - Yes
   * - Multiple searches
     - Manual
     - Built-in

----

Next Steps
==========

- :doc:`gradient_free_optimizers` - Lower-level API
- `Hyperactive Documentation <https://github.com/SimonBlanke/Hyperactive>`_
