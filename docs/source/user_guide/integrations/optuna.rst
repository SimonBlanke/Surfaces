.. _user_guide_optuna:

======
Optuna
======

Optuna is a powerful hyperparameter optimization framework with
automatic pruning, distributed execution, and visualization.

----

Installation
============

.. code-block:: bash

    pip install optuna

----

Basic Usage
===========

.. code-block:: python

    import optuna
    from surfaces.test_functions.algebraic import RastriginFunction

    func = RastriginFunction(n_dim=5)
    space = func.search_space()

    def objective(trial):
        params = {
            name: trial.suggest_float(name, values.min(), values.max())
            for name, values in space.items()
        }
        return func(params)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

----

Mixed Parameter Types
=====================

Optuna handles mixed continuous and categorical parameters:

.. code-block:: python

    from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

    func = KNeighborsClassifierFunction()

    def objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 3),
        }
        return -func(params)  # Negate for minimization

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

----

Samplers
========

Optuna supports various sampling strategies:

.. code-block:: python

    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler

    # Tree-structured Parzen Estimator (default)
    study = optuna.create_study(sampler=TPESampler())

    # CMA-ES for continuous parameters
    study = optuna.create_study(sampler=CmaEsSampler())

    # Random search baseline
    study = optuna.create_study(sampler=RandomSampler())

----

Visualization
=============

Optuna provides built-in visualization:

.. code-block:: python

    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_contour,
    )

    # Optimization history
    fig = plot_optimization_history(study)
    fig.show()

    # Parameter importance
    fig = plot_param_importances(study)
    fig.show()

    # Contour plot (2D)
    fig = plot_contour(study, params=['x0', 'x1'])
    fig.show()

----

Pruning
=======

For expensive evaluations, use pruning to stop unpromising trials:

.. code-block:: python

    from optuna.pruners import MedianPruner

    def objective(trial):
        # Report intermediate values for pruning
        for step in range(10):
            intermediate = compute_intermediate(trial, step)
            trial.report(intermediate, step)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return final_value

    study = optuna.create_study(pruner=MedianPruner())

----

Distributed Optimization
========================

Optuna supports distributed execution with a shared database:

.. code-block:: python

    # Create study with storage
    study = optuna.create_study(
        study_name='surfaces_benchmark',
        storage='sqlite:///optuna.db',
        load_if_exists=True
    )

    # Run on multiple machines/processes
    study.optimize(objective, n_trials=100)

----

Benchmarking Multiple Functions
===============================

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction, RastriginFunction, AckleyFunction

    functions = [
        ('Sphere', SphereFunction(n_dim=10)),
        ('Rastrigin', RastriginFunction(n_dim=10)),
        ('Ackley', AckleyFunction()),
    ]

    for name, func in functions:
        space = func.search_space()

        def make_objective(f, s):
            def objective(trial):
                params = {k: trial.suggest_float(k, v.min(), v.max())
                          for k, v in s.items()}
                return f(params)
            return objective

        study = optuna.create_study()
        study.optimize(make_objective(func, space), n_trials=100, show_progress_bar=True)

        print(f"{name}: {study.best_value:.6f}")

----

Next Steps
==========

- :doc:`smac` - Alternative Bayesian optimizer
- `Optuna Documentation <https://optuna.readthedocs.io/>`_ - Official docs
