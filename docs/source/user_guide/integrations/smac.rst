.. _user_guide_smac:

====
SMAC
====

SMAC (Sequential Model-based Algorithm Configuration) is a Bayesian
optimization framework designed for algorithm configuration and
hyperparameter tuning.

----

Installation
============

.. code-block:: bash

    pip install smac

----

Basic Usage
===========

.. code-block:: python

    from smac import HyperparameterOptimizationFacade, Scenario
    from ConfigSpace import ConfigurationSpace, Float

    from surfaces.test_functions.algebraic import RosenbrockFunction

    func = RosenbrockFunction(n_dim=5)
    space = func.search_space()

    # Create ConfigSpace
    cs = ConfigurationSpace()
    for name, values in space.items():
        cs.add(Float(name, bounds=(values.min(), values.max())))

    # Define target function
    def target(config, seed=0):
        params = dict(config)
        return func(params)

    # Create scenario
    scenario = Scenario(
        configspace=cs,
        n_trials=100,
        deterministic=True,
    )

    # Run optimization
    smac = HyperparameterOptimizationFacade(scenario, target)
    incumbent = smac.optimize()

    print(f"Best config: {incumbent}")
    print(f"Best value: {target(incumbent)}")

----

ConfigSpace Integration
=======================

Convert Surfaces search space to ConfigSpace:

.. code-block:: python

    from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical

    def surfaces_to_configspace(func):
        cs = ConfigurationSpace()
        space = func.search_space()

        for name, values in space.items():
            if isinstance(values[0], (int, float)):
                # Continuous parameter
                cs.add(Float(name, bounds=(float(values.min()), float(values.max()))))
            else:
                # Categorical parameter
                cs.add(Categorical(name, items=list(values)))

        return cs

    cs = surfaces_to_configspace(func)

----

Intensification
===============

SMAC uses intensification to compare configurations:

.. code-block:: python

    from smac import Scenario
    from smac.intensifier import Hyperband

    scenario = Scenario(
        configspace=cs,
        n_trials=200,
        instances=[f"instance_{i}" for i in range(5)],  # Multiple instances
    )

    # Use Hyperband for early stopping
    smac = HyperparameterOptimizationFacade(
        scenario,
        target,
        intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
    )

----

Multi-Fidelity Optimization
===========================

For expensive functions, use multi-fidelity:

.. code-block:: python

    def target(config, seed=0, budget=1.0):
        # budget controls fidelity (e.g., number of iterations)
        params = dict(config)
        return func(params)  # Adjust based on budget

    scenario = Scenario(
        configspace=cs,
        n_trials=100,
        min_budget=0.1,
        max_budget=1.0,
    )

----

Logging and Analysis
====================

SMAC provides detailed logging:

.. code-block:: python

    # Access run history
    history = smac.runhistory

    for key, value in history.items():
        config_id, instance, seed, budget = key
        cost = value.cost
        print(f"Config {config_id}: cost={cost}")

    # Get trajectory
    trajectory = smac.intensifier.trajectory
    for item in trajectory:
        print(f"Incumbent: {item.config}, cost={item.cost}")

----

Comparison with Optuna
======================

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - SMAC
     - Optuna
   * - Focus
     - Algorithm configuration
     - General HPO
   * - Surrogate
     - Random Forest
     - TPE (default)
   * - Multi-fidelity
     - Built-in
     - Via pruners
   * - Distributed
     - Via Dask
     - Via database
   * - Ease of use
     - More complex
     - Simpler API

----

Next Steps
==========

- :doc:`ray_tune` - Distributed optimization
- `SMAC Documentation <https://automl.github.io/SMAC3/>`_ - Official docs
