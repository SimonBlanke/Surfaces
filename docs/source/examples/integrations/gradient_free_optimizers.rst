.. _example_gfo:

==========================
Gradient-Free-Optimizers
==========================

Examples using Surfaces with Gradient-Free-Optimizers.

.. contents:: On this page
   :local:
   :depth: 2

----

Basic GFO Usage
===============

.. code-block:: python

    from gradient_free_optimizers import RandomSearchOptimizer
    from surfaces.test_functions import RastriginFunction

    func = RastriginFunction(n_dim=5)

    # Search space works directly - no conversion needed
    opt = RandomSearchOptimizer(func.search_space)
    opt.search(func, n_iter=100)

    print(f"Best score: {opt.best_score:.6f}")
    print(f"Best params: {opt.best_para}")

----

Different Optimizers
====================

.. code-block:: python

    from gradient_free_optimizers import (
        RandomSearchOptimizer,
        HillClimbingOptimizer,
        SimulatedAnnealingOptimizer,
        BayesianOptimizer,
        ParticleSwarmOptimizer,
    )
    from surfaces.test_functions import RastriginFunction

    func = RastriginFunction(n_dim=10)

    optimizers = [
        ('Random', RandomSearchOptimizer),
        ('HillClimbing', HillClimbingOptimizer),
        ('SimAnnealing', SimulatedAnnealingOptimizer),
        ('Bayesian', BayesianOptimizer),
        ('PSO', ParticleSwarmOptimizer),
    ]

    for name, OptClass in optimizers:
        opt = OptClass(func.search_space)
        opt.search(func, n_iter=100)
        print(f"{name:15s}: {opt.best_score:.6f}")

----

With Memory Tracking
====================

.. code-block:: python

    from gradient_free_optimizers import BayesianOptimizer
    from surfaces.test_functions import RosenbrockFunction

    func = RosenbrockFunction(n_dim=5)

    opt = BayesianOptimizer(func.search_space)
    opt.search(func, n_iter=100, memory=True)

    # Access search history
    print(f"Total evaluations: {len(opt.search_data['score'])}")
    print(f"Best score: {opt.best_score:.6f}")

    # Convergence analysis
    scores = opt.search_data['score']
    best_so_far = [min(scores[:i+1]) for i in range(len(scores))]
    print(f"Final best: {best_so_far[-1]:.6f}")
