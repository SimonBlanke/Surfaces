.. _user_guide:

==========
User Guide
==========

.. include:: _generated/diagrams/user_guide_overview.rst

----

.. tip::

   New to Surfaces? Start with :ref:`user_guide_introduction` for the fundamentals,
   then explore the :doc:`Test Functions <user_guide/test_functions/index>` to see what's available.

----

How Surfaces Works
==================

Surfaces provides a unified interface for optimization test functions.
Each function can be evaluated, returns loss or score values, and
provides its search space definition.

.. grid:: 1 1 3 3
   :gutter: 4

   .. grid-item-card:: Evaluation
      :class-card: sd-border-primary

      **Call the function**

      Evaluate at any point using dictionaries,
      keyword arguments, or numpy arrays.

   .. grid-item-card:: Metric
      :class-card: sd-border-success

      **Loss or Score**

      Results for minimization (loss) or
      maximization (score). Same function, both modes.

   .. grid-item-card:: Search Space
      :class-card: sd-border-warning

      **Parameter bounds**

      Every function provides a default search space
      with appropriate bounds.

----

Guide Sections
==============

.. grid:: 2 2 2 2
   :gutter: 4

   .. grid-item-card:: Introduction
      :link: user_guide/introduction
      :link-type: doc

      Core concepts and the unified interface.
      **Start here** if you're new to Surfaces.

   .. grid-item-card:: Test Functions
      :link: user_guide/test_functions/index
      :link-type: doc

      All function categories: Algebraic, BBOB, CEC,
      Machine Learning, and Engineering.

   .. grid-item-card:: Modifiers
      :link: user_guide/modifiers/index
      :link-type: doc

      Modify test functions with noise and
      other transformations.

   .. grid-item-card:: Presets
      :link: user_guide/presets
      :link-type: doc

      Pre-configured function collections
      for common benchmarking scenarios.

   .. grid-item-card:: Integrations
      :link: user_guide/integrations/index
      :link-type: doc

      Use Surfaces with scipy, Optuna, SMAC,
      Ray Tune, GFO, and Hyperactive.

   .. grid-item-card:: Visualization
      :link: user_guide/visualization
      :link-type: doc

      Plot function surfaces, contours,
      and optimization trajectories.

----

Quick Example
=============

.. code-block:: python

    from surfaces.test_functions import SphereFunction

    # 1. Create the function
    func = SphereFunction(n_dim=3)

    # 2. Evaluate at a point
    loss = func({"x0": 1.0, "x1": 2.0, "x2": 3.0})  # 14.0

    # 3. Get the search space
    space = func.search_space()  # {'x0': array(...), ...}

----

.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide/introduction
   user_guide/test_functions/index
   user_guide/modifiers/index
   user_guide/presets
   user_guide/integrations/index
   user_guide/visualization
   /_generated/plots/gallery
