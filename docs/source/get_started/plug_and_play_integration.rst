.. _plug_and_play_integration:

========================
Plug & Play Integration
========================

Surfaces works out of the box with popular optimization frameworks.
No adapters, no boilerplate, no hassle.

----

Supported Frameworks
====================

.. grid:: 2 2 3 3
   :gutter: 3

   .. grid-item-card:: scipy
      :link: /user_guide/integrations/scipy
      :link-type: doc

      Built-in ``to_scipy()`` method for seamless integration.

   .. grid-item-card:: Optuna
      :link: /user_guide/integrations/optuna
      :link-type: doc

      Direct use with Optuna's ``suggest_*`` API.

   .. grid-item-card:: SMAC
      :link: /user_guide/integrations/smac
      :link-type: doc

      Compatible with SMAC's configuration space.

   .. grid-item-card:: Ray Tune
      :link: /user_guide/integrations/ray_tune
      :link-type: doc

      Works with Ray Tune's search space definition.

   .. grid-item-card:: Gradient-Free-Optimizers
      :link: /user_guide/integrations/gradient_free_optimizers
      :link-type: doc

      Native integration with GFO's search space format.

   .. grid-item-card:: Hyperactive
      :link: /user_guide/integrations/hyperactive
      :link-type: doc

      Directly compatible with Hyperactive's API.

----

Quick Examples
==============

scipy
-----

Every Surfaces function has a built-in ``to_scipy()`` method:

.. code-block:: python

    from surfaces.test_functions.algebraic import RosenbrockFunction
    from scipy.optimize import minimize

    func = RosenbrockFunction(n_dim=5)

    # Convert to scipy format
    objective, bounds, x0 = func.to_scipy()

    # Run optimization
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    print(f"Minimum: {result.fun:.6f}")

Optuna
------

Use Surfaces functions directly in Optuna objectives:

.. code-block:: python

    import optuna
    from surfaces.test_functions.algebraic import AckleyFunction

    func = AckleyFunction(n_dim=3)
    space = func.search_space()

    def objective(trial):
        params = {
            name: trial.suggest_float(name, values.min(), values.max())
            for name, values in space.items()
        }
        return func(params)

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

Gradient-Free-Optimizers
------------------------

Surfaces search spaces work directly with GFO:

.. code-block:: python

    from gradient_free_optimizers import RandomSearchOptimizer
    from surfaces.test_functions.algebraic import RastriginFunction

    func = RastriginFunction(n_dim=5)

    opt = RandomSearchOptimizer(func.search_space())
    opt.search(func, n_iter=100)

    print(f"Best score: {opt.best_score}")

----

The Unified Interface
=====================

What makes Surfaces truly plug & play is the unified interface.
Every test function provides:

1. **Callable evaluation**: ``func(params)`` works with dictionaries
2. **Search space**: ``func.search_space()`` returns parameter bounds
3. **scipy conversion**: ``func.to_scipy()`` for scipy optimizers
4. **Loss and score**: Both minimization and maximization supported

This consistency means you write the integration code once and it
works with all 100+ functions in the library.

----

Next Steps
==========

- :doc:`/user_guide/integrations/index` - Detailed integration guides for each framework
- :doc:`/examples/integrations/index` - Complete integration examples
