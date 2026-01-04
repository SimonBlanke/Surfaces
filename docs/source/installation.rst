.. _installation:

============
Installation
============

Surfaces is designed to be lightweight. The core installation requires
only **numpy** as a dependency.

----

Core Installation
=================

.. code-block:: bash

    pip install surfaces

This gives you:

- All **Algebraic** test functions (1D, 2D, N-D)
- All **BBOB** functions (Black-Box Optimization Benchmarking)
- All **CEC** functions (Competition on Evolutionary Computation)
- All **Engineering** design problems

**Requirements:**

- Python |min_python| or higher
- NumPy >= 1.18.1

That's it. No heavy dependencies, fast installation, CI/CD friendly.

----

Optional Features
=================

Add features only when you need them:

.. grid:: 2 2 2 2
   :gutter: 4

   .. grid-item-card:: CEC Functions
      :link: installation/cec
      :link-type: doc

      Additional setup for CEC benchmark suites.
      Some CEC functions require extra configuration.

      .. code-block:: bash

          pip install surfaces[cec]

   .. grid-item-card:: Machine Learning
      :link: installation/machine_learning
      :link-type: doc

      ML-based test functions using scikit-learn,
      XGBoost, and other ML libraries.

      .. code-block:: bash

          pip install surfaces[ml]

   .. grid-item-card:: Surrogates
      :link: installation/surrogates
      :link-type: doc

      Pre-trained surrogate models for fast
      evaluation of expensive functions.

      .. code-block:: bash

          pip install surfaces[surrogates]

   .. grid-item-card:: Visualization
      :link: installation/visualization
      :link-type: doc

      Surface plots, contour plots, and
      interactive visualizations.

      .. code-block:: bash

          pip install surfaces[viz]

----

Full Installation
=================

Install everything at once:

.. code-block:: bash

    pip install surfaces[full]

This includes all optional features: ML functions, surrogates, and visualization.

----

Installing from Source
======================

For the latest development version:

.. code-block:: bash

    git clone https://github.com/SimonBlanke/Surfaces.git
    cd Surfaces
    pip install -e .

For development with all dependencies:

.. code-block:: bash

    pip install -e ".[dev,test]"

----

Verifying Installation
======================

.. code-block:: python

    import surfaces
    print(f"Surfaces version: {surfaces.__version__}")

    # Test a simple function
    from surfaces.test_functions import SphereFunction
    func = SphereFunction(n_dim=2)
    result = func({"x0": 0.0, "x1": 0.0})
    print(f"Sphere(0, 0) = {result}")  # Should print 0.0

----

Platform Support
================

Surfaces is tested on:

- Linux (Ubuntu)
- macOS
- Windows

----

Troubleshooting
===============

Import Errors
-------------

If you get import errors for specific features:

.. code-block:: bash

    # For ML functions
    pip install surfaces[ml]

    # For visualization
    pip install surfaces[viz]

    # For surrogates
    pip install surfaces[surrogates]

Version Conflicts
-----------------

Try a fresh virtual environment:

.. code-block:: bash

    python -m venv surfaces-env
    source surfaces-env/bin/activate  # Windows: surfaces-env\Scripts\activate
    pip install surfaces

Getting Help
------------

1. Check the :doc:`faq` for common problems
2. Search `existing issues <https://github.com/SimonBlanke/Surfaces/issues>`_
3. Open a new issue on GitHub


.. toctree::
   :maxdepth: 1
   :hidden:

   installation/cec
   installation/machine_learning
   installation/surrogates
   installation/visualization
