.. _user_guide_integrations:

============
Integrations
============

Surfaces integrates seamlessly with popular optimization frameworks.
Use your favorite optimizer with any Surfaces test function.

----

Supported Frameworks
====================

.. grid:: 2 2 3 3
   :gutter: 3

   .. grid-item-card:: scipy
      :link: scipy
      :link-type: doc

      Scientific Python's optimization module.
      Built-in ``to_scipy()`` conversion.

   .. grid-item-card:: Optuna
      :link: optuna
      :link-type: doc

      Hyperparameter optimization framework.
      Works with ``suggest_*`` API.

   .. grid-item-card:: SMAC
      :link: smac
      :link-type: doc

      Sequential Model-based Algorithm Configuration.
      ConfigSpace integration.

   .. grid-item-card:: Ray Tune
      :link: ray_tune
      :link-type: doc

      Distributed hyperparameter tuning.
      Scalable experimentation.

   .. grid-item-card:: Gradient-Free-Optimizers
      :link: gradient_free_optimizers
      :link-type: doc

      Simple gradient-free optimization.
      Native search space compatibility.

   .. grid-item-card:: Hyperactive
      :link: hyperactive
      :link-type: doc

      Advanced optimization toolkit.
      Direct integration.

----

Quick Comparison
================

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Framework
     - Type
     - Scale
     - Best For
   * - scipy
     - Classical
     - Single machine
     - Quick prototyping, gradient methods
   * - Optuna
     - Bayesian
     - Distributed
     - HPO, pruning, visualization
   * - SMAC
     - Bayesian
     - Single machine
     - Algorithm configuration, research
   * - Ray Tune
     - Various
     - Distributed
     - Large-scale, multi-GPU
   * - GFO
     - Gradient-free
     - Single machine
     - Simple API, exploration
   * - Hyperactive
     - Meta-heuristic
     - Single machine
     - Advanced search strategies

----

Common Pattern
==============

All integrations follow a similar pattern:

1. Create a Surfaces test function
2. Extract the search space
3. Define an objective that calls the test function
4. Run the optimizer

.. code-block:: python

    from surfaces.test_functions.algebraic import RastriginFunction

    # 1. Create function
    func = RastriginFunction(n_dim=5)

    # 2. Get search space
    space = func.search_space()

    # 3. Define objective
    def objective(params):
        return func(params)

    # 4. Run optimizer (framework-specific)
    # ...

----

Choosing a Framework
====================

**For quick experiments:**

Use scipy or GFO. Minimal setup, fast iteration.

**For serious benchmarking:**

Use Optuna or SMAC. Better optimization, logging, visualization.

**For large-scale studies:**

Use Ray Tune. Distributed execution, checkpointing, scalability.

**For research:**

Use SMAC or custom. Fine-grained control, reproducibility.

----

.. toctree::
   :maxdepth: 1
   :hidden:

   scipy
   optuna
   smac
   ray_tune
   gradient_free_optimizers
   hyperactive
