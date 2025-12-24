.. _user_guide_optimizer_compatibility:

======================
Optimizer Frameworks
======================

Surfaces test functions work with all major optimization frameworks.
This guide shows how to integrate Surfaces with each framework.

Compatibility Overview
======================

Surfaces functions accept multiple input formats, making them compatible
with virtually any optimization framework:

.. list-table:: Supported Frameworks
   :header-rows: 1
   :widths: 20 15 15 50

   * - Framework
     - Input Format
     - Status
     - Notes
   * - **Gradient-Free-Optimizers**
     - dict
     - Tested
     - Native integration
   * - **Hyperactive**
     - dict
     - Tested
     - Native integration
   * - **scipy.optimize**
     - array
     - Tested
     - See :ref:`user_guide_scipy_integration`
   * - **Optuna**
     - dict
     - Tested
     - Via suggest_* methods
   * - **SMAC**
     - dict-like
     - Tested
     - ConfigSpace Configuration objects
   * - **Nevergrad**
     - array or dict
     - Tested
     - Supports both parametrizations
   * - **Bayesian Optimization**
     - kwargs
     - Tested
     - Passes parameters as keyword arguments
   * - **Hyperopt**
     - dict
     - Tested
     - Native dict support
   * - **scikit-optimize**
     - list
     - Tested
     - Passes parameters as list
   * - **Ray Tune**
     - dict
     - Tested
     - Via config dict

Input Formats
=============

Surfaces functions accept these input formats:

.. code-block:: python

    from surfaces.test_functions import SphereFunction

    func = SphereFunction(n_dim=2)

    # All of these produce the same result:
    func({"x0": 1.0, "x1": 2.0})      # dict
    func([1.0, 2.0])                   # list
    func((1.0, 2.0))                   # tuple
    func(np.array([1.0, 2.0]))         # numpy array
    func(x0=1.0, x1=2.0)               # kwargs

Framework Examples
==================

Gradient-Free-Optimizers
------------------------

.. code-block:: python

    from gradient_free_optimizers import RandomSearchOptimizer
    from surfaces.test_functions import SphereFunction
    import numpy as np

    func = SphereFunction(n_dim=2)

    search_space = {
        "x0": np.arange(-5, 5, 0.1),
        "x1": np.arange(-5, 5, 0.1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(func, n_iter=100)

Optuna
------

.. code-block:: python

    import optuna
    from surfaces.test_functions import SphereFunction

    func = SphereFunction(n_dim=2)

    def objective(trial):
        x0 = trial.suggest_float("x0", -5, 5)
        x1 = trial.suggest_float("x1", -5, 5)
        return func({"x0": x0, "x1": x1})

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

SMAC
----

.. code-block:: python

    from ConfigSpace import ConfigurationSpace, Float
    from smac import HyperparameterOptimizationFacade, Scenario
    from surfaces.test_functions import SphereFunction

    func = SphereFunction(n_dim=2)

    def objective(config, seed=0):
        return func({"x0": config["x0"], "x1": config["x1"]})

    configspace = ConfigurationSpace(seed=42)
    configspace.add(Float("x0", (-5.0, 5.0)))
    configspace.add(Float("x1", (-5.0, 5.0)))

    scenario = Scenario(configspace, n_trials=100)
    smac = HyperparameterOptimizationFacade(scenario, objective)
    incumbent = smac.optimize()

Nevergrad
---------

Nevergrad supports both array-based and dict-based parametrizations:

**Array-based:**

.. code-block:: python

    import nevergrad as ng
    from surfaces.test_functions import SphereFunction

    func = SphereFunction(n_dim=2)

    parametrization = ng.p.Array(shape=(2,), lower=-5, upper=5)
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=100)
    recommendation = optimizer.minimize(func)

    print(f"Best params: {recommendation.value}")

**Dict-based:**

.. code-block:: python

    import nevergrad as ng
    from surfaces.test_functions import SphereFunction

    func = SphereFunction(n_dim=2)

    parametrization = ng.p.Instrumentation(
        x0=ng.p.Scalar(lower=-5, upper=5),
        x1=ng.p.Scalar(lower=-5, upper=5),
    )

    def objective(x0, x1):
        return func({"x0": x0, "x1": x1})

    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=100)
    recommendation = optimizer.minimize(objective)

Bayesian Optimization
---------------------

.. code-block:: python

    from bayes_opt import BayesianOptimization
    from surfaces.test_functions import SphereFunction

    func = SphereFunction(n_dim=2)

    # bayesian-optimization maximizes, so negate for minimization
    def objective(x0, x1):
        return -func({"x0": x0, "x1": x1})

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={"x0": (-5, 5), "x1": (-5, 5)},
    )
    optimizer.maximize(init_points=5, n_iter=50)

    print(f"Best params: {optimizer.max['params']}")
    print(f"Best value: {-optimizer.max['target']}")

Hyperopt
--------

.. code-block:: python

    from hyperopt import fmin, tpe, hp
    from surfaces.test_functions import SphereFunction

    func = SphereFunction(n_dim=2)

    space = {
        "x0": hp.uniform("x0", -5, 5),
        "x1": hp.uniform("x1", -5, 5),
    }

    best = fmin(
        fn=func,  # Surfaces accepts dict directly
        space=space,
        algo=tpe.suggest,
        max_evals=100,
    )

    print(f"Best params: {best}")

scikit-optimize
---------------

.. code-block:: python

    from skopt import gp_minimize
    from surfaces.test_functions import SphereFunction

    func = SphereFunction(n_dim=2)

    # skopt passes parameters as a list
    result = gp_minimize(
        func,  # Surfaces accepts list directly
        [(-5.0, 5.0), (-5.0, 5.0)],
        n_calls=50,
    )

    print(f"Best params: {result.x}")
    print(f"Best value: {result.fun}")

Ray Tune
--------

.. code-block:: python

    from ray import tune
    from surfaces.test_functions import SphereFunction

    func = SphereFunction(n_dim=2)

    def objective(config):
        result = func({"x0": config["x0"], "x1": config["x1"]})
        return {"loss": result}

    tuner = tune.Tuner(
        objective,
        param_space={
            "x0": tune.uniform(-5, 5),
            "x1": tune.uniform(-5, 5),
        },
        tune_config=tune.TuneConfig(
            num_samples=100,
            metric="loss",
            mode="min",
        ),
    )
    results = tuner.fit()

scipy.optimize
--------------

See :ref:`user_guide_scipy_integration` for detailed scipy integration.

.. code-block:: python

    from scipy.optimize import minimize
    from surfaces.test_functions import SphereFunction

    func = SphereFunction(n_dim=2)

    # Surfaces accepts numpy arrays directly
    result = minimize(
        func,
        x0=[1.0, 1.0],
        bounds=[(-5, 5), (-5, 5)],
        method="L-BFGS-B",
    )

    print(f"Best params: {result.x}")
    print(f"Best value: {result.fun}")

ML Functions with Mixed Parameters
==================================

For ML functions with categorical parameters, use frameworks that support
mixed parameter types:

.. code-block:: python

    import optuna
    from surfaces.test_functions import KNeighborsClassifierFunction
    from surfaces.test_functions.machine_learning.tabular.classification.datasets import iris_data

    func = KNeighborsClassifierFunction()

    def objective(trial):
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree"]),
            "cv": trial.suggest_int("cv", 2, 10),
            "dataset": iris_data,
        }
        return -func(params)  # Negate because we want to maximize accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

Memory Feature
==============

When running many evaluations, enable memory caching to skip redundant
computations:

.. code-block:: python

    from surfaces.test_functions import SphereFunction

    # Enable memory caching
    func = SphereFunction(n_dim=2, memory=True)

    # First call evaluates
    result1 = func([0.0, 0.0])

    # Second call returns cached value (no computation)
    result2 = func([0.0, 0.0])

    # Access the cache
    print(func._memory_cache)  # {(0.0, 0.0): 0.0}

This is especially useful with grid search or when the optimizer
may revisit the same points.
