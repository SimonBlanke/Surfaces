.. _user_guide_gfo:

==========================
Gradient-Free-Optimizers
==========================

Gradient-Free-Optimizers (GFO) is a simple optimization library with
a clean API. Surfaces search spaces work directly with GFO.

----

Installation
============

.. code-block:: bash

    pip install gradient-free-optimizers

----

Basic Usage
===========

.. code-block:: python

    from gradient_free_optimizers import RandomSearchOptimizer
    from surfaces.test_functions import RastriginFunction

    func = RastriginFunction(n_dim=5)

    # Search space is directly compatible
    opt = RandomSearchOptimizer(func.search_space())
    opt.search(func, n_iter=100)

    print(f"Best score: {opt.best_score}")
    print(f"Best params: {opt.best_para}")

That's it. No conversion needed.

----

Available Optimizers
====================

Local Search
------------

.. code-block:: python

    from gradient_free_optimizers import (
        HillClimbingOptimizer,
        StochasticHillClimbingOptimizer,
        SimulatedAnnealingOptimizer,
    )

    # Hill Climbing
    opt = HillClimbingOptimizer(func.search_space())
    opt.search(func, n_iter=100)

    # Simulated Annealing
    opt = SimulatedAnnealingOptimizer(func.search_space())
    opt.search(func, n_iter=100)

Population-Based
----------------

.. code-block:: python

    from gradient_free_optimizers import (
        ParticleSwarmOptimizer,
        EvolutionStrategyOptimizer,
    )

    # Particle Swarm
    opt = ParticleSwarmOptimizer(func.search_space(), population=20)
    opt.search(func, n_iter=100)

    # Evolution Strategy
    opt = EvolutionStrategyOptimizer(func.search_space(), population=20)
    opt.search(func, n_iter=100)

Bayesian Optimization
---------------------

.. code-block:: python

    from gradient_free_optimizers import BayesianOptimizer

    opt = BayesianOptimizer(func.search_space())
    opt.search(func, n_iter=100)

----

Optimization Parameters
=======================

.. code-block:: python

    # Control exploration vs exploitation
    opt = SimulatedAnnealingOptimizer(
        func.search_space(),
        start_temp=10.0,
        annealing_rate=0.97,
    )

    # Warm start from previous results
    opt = BayesianOptimizer(func.search_space())
    opt.search(func, n_iter=50)

    # Continue optimization
    opt.search(func, n_iter=50)  # Continues from previous state

----

Memory and Verbosity
====================

.. code-block:: python

    # Track all evaluations
    opt = RandomSearchOptimizer(func.search_space())
    opt.search(func, n_iter=100, memory=True)

    # Access search history
    for params, score in zip(opt.search_data["params"], opt.search_data["scores"]):
        print(f"{params}: {score}")

    # Progress bar
    opt.search(func, n_iter=100, verbosity=["progress_bar"])

----

Benchmarking Example
====================

Compare optimizers on Surfaces functions:

.. code-block:: python

    from gradient_free_optimizers import (
        RandomSearchOptimizer,
        HillClimbingOptimizer,
        BayesianOptimizer,
        ParticleSwarmOptimizer,
    )
    from surfaces.test_functions import RastriginFunction

    func = RastriginFunction(n_dim=10)

    optimizers = [
        ("Random", RandomSearchOptimizer),
        ("HillClimbing", HillClimbingOptimizer),
        ("Bayesian", BayesianOptimizer),
        ("PSO", ParticleSwarmOptimizer),
    ]

    for name, OptClass in optimizers:
        opt = OptClass(func.search_space())
        opt.search(func, n_iter=100)
        print(f"{name}: {opt.best_score:.6f}")

----

Why GFO?
========

- **Simple**: Minimal boilerplate
- **Compatible**: Surfaces search spaces work directly
- **Variety**: Many optimizer types available
- **Lightweight**: Few dependencies

For more features (distributed, pruning), use Optuna or Ray Tune.

----

Next Steps
==========

- :doc:`hyperactive` - Advanced GFO-based toolkit
- `GFO Documentation <https://github.com/SimonBlanke/Gradient-Free-Optimizers>`_
