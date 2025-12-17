.. _installation:

============
Installation
============

This page covers various ways to install Surfaces and its dependencies.

Requirements
============

Surfaces requires:

- Python |min_python| or higher
- NumPy >= 1.18.1
- pandas
- matplotlib
- plotly
- scikit-learn
- search-data-collector >= 0.6.0

Installing from PyPI
====================

The recommended way to install Surfaces is via pip:

.. code-block:: bash

    pip install surfaces

This will install Surfaces and all required dependencies.

Installing with Extras
======================

For development or testing, you can install additional dependencies:

.. code-block:: bash

    # Development dependencies
    pip install surfaces[dev]

    # Test dependencies
    pip install surfaces[test]

    # All extras
    pip install surfaces[dev,test]

Installing from Source
======================

To install the latest development version from GitHub:

.. code-block:: bash

    git clone https://github.com/SimonBlanke/Surfaces.git
    cd Surfaces
    pip install -e .

For development with all dependencies:

.. code-block:: bash

    pip install -e ".[dev,test]"

Verifying Installation
======================

You can verify that Surfaces is installed correctly:

.. code-block:: python

    import surfaces
    print(surfaces.__version__)

    # Test a simple function
    from surfaces.test_functions import SphereFunction
    func = SphereFunction(n_dim=2)
    result = func({"x0": 0.0, "x1": 0.0})
    print(f"Sphere(0, 0) = {result}")  # Should print 0.0

Platform Support
================

Surfaces is tested on:

- Linux (Ubuntu)
- macOS
- Windows

Surfaces should work on any platform where Python and its dependencies are available.

Dependency Notes
================

NumPy
-----

Surfaces uses NumPy for numerical operations. It is compatible with NumPy 1.x and 2.x.

scikit-learn
------------

The machine learning test functions require scikit-learn for model training.
Any recent version of scikit-learn should work.

Plotly
------

The visualization features use Plotly for interactive plots. Plotly is optional
for core functionality but required for visualization.

Troubleshooting
===============

Import Errors
-------------

If you get import errors, make sure all dependencies are installed:

.. code-block:: bash

    pip install numpy pandas matplotlib plotly scikit-learn

Version Conflicts
-----------------

If you encounter version conflicts, try creating a fresh virtual environment:

.. code-block:: bash

    python -m venv surfaces-env
    source surfaces-env/bin/activate  # On Windows: surfaces-env\Scripts\activate
    pip install surfaces

Getting Help
------------

If you encounter issues:

1. Check the :doc:`faq` for common problems
2. Search `existing issues <https://github.com/SimonBlanke/Surfaces/issues>`_
3. Open a new issue on GitHub
