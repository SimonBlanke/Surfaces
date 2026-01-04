.. _example_hyperactive:

==========
Hyperactive
==========

Examples using Surfaces with Hyperactive.

.. contents:: On this page
   :local:
   :depth: 2

----

Basic Hyperactive Usage
=======================

.. code-block:: python

    from hyperactive import Hyperactive
    from surfaces.test_functions import RastriginFunction

    func = RastriginFunction(n_dim=5)

    hyper = Hyperactive()
    hyper.add_search(func, func.search_space(), n_iter=100)
    hyper.run()

    print(f"Best score: {hyper.best_score(func):.6f}")
    print(f"Best params: {hyper.best_para(func)}")

----

Parallel Optimization
=====================

.. code-block:: python

    from hyperactive import Hyperactive
    from surfaces.test_functions import SphereFunction, RastriginFunction, RosenbrockFunction

    sphere = SphereFunction(n_dim=10)
    rastrigin = RastriginFunction(n_dim=10)
    rosenbrock = RosenbrockFunction(n_dim=10)

    hyper = Hyperactive()
    hyper.add_search(sphere, sphere.search_space(), n_iter=100)
    hyper.add_search(rastrigin, rastrigin.search_space(), n_iter=100)
    hyper.add_search(rosenbrock, rosenbrock.search_space(), n_iter=100)

    # Run all in parallel
    hyper.run(max_workers=3)

    for func in [sphere, rastrigin, rosenbrock]:
        print(f"{func.__class__.__name__}: {hyper.best_score(func):.6f}")

----

With Different Optimizers
=========================

.. code-block:: python

    from hyperactive import Hyperactive
    from hyperactive.optimizers import BayesianOptimizer, ParticleSwarmOptimizer
    from surfaces.test_functions import RastriginFunction

    func = RastriginFunction(n_dim=10)

    hyper = Hyperactive()

    # Compare different optimizers
    hyper.add_search(
        func,
        func.search_space(),
        optimizer=BayesianOptimizer(),
        n_iter=100,
    )

    hyper.run()
    print(f"Bayesian: {hyper.best_score(func):.6f}")
