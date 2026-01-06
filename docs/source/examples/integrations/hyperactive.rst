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

    from hyperactive.opt.gfo import HillClimbing
    from surfaces.test_functions.algebraic import RastriginFunction

    func = RastriginFunction(n_dim=5)

    optimizer = HillClimbing(
        search_space=func.search_space,
        n_iter=100,
        experiment=func,
    )
    best_params = optimizer.solve()

    print(f"Best params: {best_params}")
    print(f"Best score: {func(best_params):.6f}")

----

Different Optimizers
====================

.. code-block:: python

    from hyperactive.opt.gfo import (
        HillClimbing,
        RandomSearch,
        SimulatedAnnealing,
    )
    from surfaces.test_functions.algebraic import RastriginFunction

    func = RastriginFunction(n_dim=5)

    optimizers = [
        ('HillClimbing', HillClimbing),
        ('RandomSearch', RandomSearch),
        ('SimAnnealing', SimulatedAnnealing),
    ]

    for name, OptClass in optimizers:
        optimizer = OptClass(
            search_space=func.search_space,
            n_iter=100,
            experiment=func,
        )
        best = optimizer.solve()
        print(f"{name}: {func(best):.6f}")

----

.. note::

    For the latest Hyperactive API, see the
    `Hyperactive documentation <https://github.com/SimonBlanke/Hyperactive>`_.
