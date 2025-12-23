.. _user_guide:

==========
User Guide
==========

Master Surfaces' test functions. This guide covers core concepts,
function categories, and integration patterns.

.. tip::

   New to Surfaces? Start with :ref:`user_guide_introduction` for the fundamentals,
   then explore :ref:`user_guide_mathematical` to see available functions.

----

How Surfaces Works
------------------

Surfaces provides a unified interface for optimization test functions.
Each function can be evaluated, returns loss or score values, and
provides its search space definition.

Core Concepts
-------------

Every test function in Surfaces involves three components:

.. grid:: 1 1 3 3
   :gutter: 4

   .. grid-item-card:: Evaluation
      :class-card: sd-border-primary

      **Call the function**
      ^^^
      Evaluate the function at any point using dictionaries,
      keyword arguments, or numpy arrays.

      +++
      :doc:`Learn more <user_guide/introduction>`

   .. grid-item-card:: Metric
      :class-card: sd-border-success

      **Loss or Score**
      ^^^
      Get results for minimization (loss) or maximization (score).
      The same function supports both paradigms.

      +++
      :doc:`Learn more <user_guide/introduction>`

   .. grid-item-card:: Search Space
      :class-card: sd-border-warning

      **Parameter bounds**
      ^^^
      Every function provides a default search space with
      appropriate bounds for optimization.

      +++
      :doc:`Learn more <user_guide/introduction>`

----

Quick Example
-------------

.. code-block:: python

    from surfaces.test_functions import SphereFunction

    # 1. Create the function
    func = SphereFunction(n_dim=3)

    # 2. Evaluate at a point
    loss = func({"x0": 1.0, "x1": 2.0, "x2": 3.0})  # 14.0

    # 3. Get the search space
    space = func.search_space()  # {'x0': array(...), ...}

----

Guide Sections
--------------

.. grid:: 2 2 3 3
   :gutter: 3

   .. grid-item-card:: Introduction
      :link: user_guide/introduction
      :link-type: doc

      Core concepts and interface.
      **Start here** if you're new.

   .. grid-item-card:: Mathematical Functions
      :link: user_guide/mathematical
      :link-type: doc

      Classic test functions from the
      optimization literature.

   .. grid-item-card:: ML Functions
      :link: user_guide/machine_learning
      :link-type: doc

      Test functions based on
      machine learning models.

   .. grid-item-card:: scipy Integration
      :link: user_guide/scipy_integration
      :link-type: doc

      Use Surfaces with scipy.optimize
      for seamless optimization.

   .. grid-item-card:: Optimizer Frameworks
      :link: user_guide/optimizer_compatibility
      :link-type: doc

      Integrate with Optuna, SMAC,
      Ray Tune, and more.

   .. grid-item-card:: Visualization
      :link: user_guide/visualization
      :link-type: doc

      Plot function surfaces
      and contours.


.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide/introduction
   user_guide/mathematical
   user_guide/machine_learning
   user_guide/scipy_integration
   user_guide/optimizer_compatibility
   user_guide/visualization
