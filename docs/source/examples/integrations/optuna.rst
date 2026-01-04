.. _example_optuna:

======
Optuna
======

Examples using Surfaces with Optuna.

.. contents:: On this page
   :local:
   :depth: 2

----

Basic Optuna Usage
==================

.. code-block:: python

    import optuna
    from surfaces.test_functions import RastriginFunction

    func = RastriginFunction(n_dim=5)
    space = func.search_space()

    def objective(trial):
        params = {
            name: trial.suggest_float(name, values.min(), values.max())
            for name, values in space.items()
        }
        return func(params)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

----

With Categorical Parameters
===========================

.. code-block:: python

    import optuna
    from surfaces.test_functions import KNeighborsClassifierFunction

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

    print(f"Best accuracy: {-study.best_value:.4f}")

----

Benchmarking Multiple Functions
===============================

.. code-block:: python

    """Benchmark Optuna on multiple Surfaces functions."""

    import optuna
    from surfaces.test_functions import SphereFunction, RastriginFunction, AckleyFunction

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    functions = [
        ('Sphere', SphereFunction(n_dim=10)),
        ('Rastrigin', RastriginFunction(n_dim=10)),
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
        study.optimize(make_objective(func, space), n_trials=100)

        print(f"{name}: {study.best_value:.6f}")
