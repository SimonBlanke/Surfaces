.. _user_guide_ray_tune:

========
Ray Tune
========

Ray Tune is a scalable hyperparameter tuning library that supports
distributed execution across multiple machines and GPUs.

----

Installation
============

.. code-block:: bash

    pip install "ray[tune]"

----

Basic Usage
===========

.. code-block:: python

    from ray import tune
    from surfaces.test_functions.algebraic import RastriginFunction

    func = RastriginFunction(n_dim=5)
    space = func.search_space()

    # Define search space
    config = {
        name: tune.uniform(values.min(), values.max())
        for name, values in space.items()
    }

    # Define trainable
    def trainable(config):
        result = func(config)
        return {"loss": result}

    # Run tuning
    analysis = tune.run(
        trainable,
        config=config,
        num_samples=100,
        metric="loss",
        mode="min",
    )

    print(f"Best config: {analysis.best_config}")
    print(f"Best loss: {analysis.best_result['loss']}")

----

Search Algorithms
=================

Ray Tune supports various search algorithms:

.. code-block:: python

    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search.bayesopt import BayesOptSearch

    # Optuna integration
    analysis = tune.run(
        trainable,
        config=config,
        search_alg=OptunaSearch(),
        num_samples=100,
    )

    # Bayesian optimization
    analysis = tune.run(
        trainable,
        config=config,
        search_alg=BayesOptSearch(),
        num_samples=100,
    )

----

Schedulers
==========

Use schedulers for early stopping:

.. code-block:: python

    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

    # ASHA (Asynchronous Successive Halving)
    scheduler = ASHAScheduler(
        max_t=100,
        grace_period=10,
        reduction_factor=2,
    )

    analysis = tune.run(
        trainable,
        config=config,
        scheduler=scheduler,
        num_samples=100,
    )

----

Distributed Execution
=====================

Ray Tune scales across machines:

.. code-block:: python

    import ray

    # Connect to Ray cluster
    ray.init(address="auto")

    # Resources per trial
    analysis = tune.run(
        trainable,
        config=config,
        num_samples=1000,
        resources_per_trial={"cpu": 2, "gpu": 0.5},
    )

----

Checkpointing
=============

For long-running experiments:

.. code-block:: python

    from ray.air import session
    from ray.air.checkpoint import Checkpoint

    def trainable(config):
        # Load checkpoint if exists
        checkpoint = session.get_checkpoint()
        if checkpoint:
            state = checkpoint.to_dict()
            start_iter = state["iteration"]
        else:
            start_iter = 0

        for i in range(start_iter, 100):
            result = func(config)

            # Save checkpoint
            checkpoint = Checkpoint.from_dict({"iteration": i})
            session.report({"loss": result}, checkpoint=checkpoint)

----

Benchmarking Example
====================

Compare search algorithms:

.. code-block:: python

    from ray.tune.search import BasicVariantGenerator
    from ray.tune.search.optuna import OptunaSearch

    func = RastriginFunction(n_dim=10)
    space = func.search_space()

    config = {name: tune.uniform(v.min(), v.max()) for name, v in space.items()}

    algorithms = [
        ("Random", BasicVariantGenerator()),
        ("Optuna", OptunaSearch()),
    ]

    for name, search_alg in algorithms:
        analysis = tune.run(
            lambda c: {"loss": func(c)},
            config=config,
            search_alg=search_alg,
            num_samples=100,
            verbose=0,
        )
        print(f"{name}: {analysis.best_result['loss']:.6f}")

----

When to Use Ray Tune
====================

- **Large-scale experiments**: Thousands of trials
- **Multi-GPU**: Distribute across GPUs
- **Production**: Checkpointing, fault tolerance
- **Algorithm comparison**: Multiple search algorithms

For simple experiments, Optuna or GFO may be easier.

----

Next Steps
==========

- :doc:`gradient_free_optimizers` - Simpler alternative
- `Ray Tune Documentation <https://docs.ray.io/en/latest/tune/>`_ - Official docs
