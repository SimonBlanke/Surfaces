.. _examples:

========
Examples
========

.. include:: _generated/diagrams/examples_overview.rst

----

Code examples demonstrating how to use Surfaces for optimization benchmarking.
All examples are tested and can be run as standalone scripts.

----

Example Categories
==================

.. grid:: 2 2 2 2
   :gutter: 4

   .. grid-item-card:: Getting Started
      :link: examples/getting_started/index
      :link-type: doc

      First steps with Surfaces. Basic usage,
      input formats, and core concepts.

   .. grid-item-card:: Test Functions
      :link: examples/test_functions/index
      :link-type: doc

      Working with different function categories:
      Algebraic, BBOB, CEC, ML, Engineering.

   .. grid-item-card:: Integrations
      :link: examples/integrations/index
      :link-type: doc

      Using Surfaces with scipy, Optuna, SMAC,
      Ray Tune, GFO, and Hyperactive.

   .. grid-item-card:: Visualization
      :link: examples/visualization_examples/index
      :link-type: doc

      Creating surface plots, contour plots,
      and optimization trajectories.

----

Quick Links
===========

**Most Popular:**

- :doc:`examples/getting_started/basic_usage` - Your first test function
- :doc:`examples/integrations/scipy` - scipy.optimize integration
- :doc:`examples/integrations/optuna` - Optuna hyperparameter tuning

----

Running Examples
================

All examples can be run as standalone Python scripts:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/SimonBlanke/Surfaces.git
    cd Surfaces

    # Install with examples dependencies
    pip install -e ".[examples]"

    # Run an example
    python docs/examples/getting_started/basic_usage.py

----

.. toctree::
   :maxdepth: 2
   :hidden:

   examples/getting_started/index
   examples/test_functions/index
   examples/integrations/index
   examples/visualization_examples/index
