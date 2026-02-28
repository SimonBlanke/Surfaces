.. _get_involved:

============
Get Involved
============

Surfaces is an open-source project and we welcome contributions from the community.

Ways to Contribute
==================

There are many ways to help improve Surfaces:

Report Bugs
-----------

Found a bug? Please open an issue on GitHub:

1. Search `existing issues <https://github.com/SimonBlanke/Surfaces/issues>`_ first
2. If not found, `open a new issue <https://github.com/SimonBlanke/Surfaces/issues/new>`_
3. Include:

   - Surfaces version
   - Python version
   - Operating system
   - Minimal code to reproduce
   - Full error traceback

Suggest Features
----------------

Have an idea for a new feature or test function?

1. Check if it's already been suggested
2. Open an issue describing:

   - What you'd like to see
   - Why it would be useful
   - Example use cases

Improve Documentation
---------------------

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples
- Improve API documentation
- Write tutorials

Submit Code
-----------

Ready to contribute code?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

Development Setup
=================

To set up a development environment:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/SimonBlanke/Surfaces.git
    cd Surfaces

    # Create a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install in editable mode with dev dependencies
    pip install -e ".[dev,test]"

Running Tests
=============

Run the test suite:

.. code-block:: bash

    # Run all tests
    make test

    # Run just pytest
    make py-test

    # Run linting
    flake8

Code Style
==========

Please follow these guidelines:

- Use `black <https://black.readthedocs.io/>`_ for code formatting
- Follow PEP 8 style guidelines
- Add docstrings to all public functions and classes
- Include type hints where appropriate

Adding Test Functions
=====================

Surfaces uses the **template method pattern**: public methods are fixed in
base classes and delegate to private override points. See
:ref:`developer_template_method` for the full architecture reference.

To add a new test function:

1. Create a new file in the appropriate category
2. Inherit from the appropriate base class
3. Implement the required override point(s) for that base class
4. Add to ``__init__.py`` exports
5. Write tests

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Base Class
     - Required Overrides
     - Search Space
   * - ``AlgebraicFunction``
     - ``_objective(params)``
     - Automatic from ``default_bounds`` + ``n_dim``
   * - ``EngineeringFunction``
     - ``_raw_objective(params)``, ``_constraints(params)``
     - Automatic from ``variable_names`` + ``variable_bounds``
   * - ``MachineLearningFunction``
     - ``_ml_objective(params)``, ``_default_search_space()``
     - Built from ``para_names`` + ``*_default`` attrs
   * - ``SimulationFunction``
     - ``_setup_simulation()``, ``_run_simulation(params)``, ``_extract_objective(result)``, ``_default_search_space()``
     - Explicit dict per function

Example: adding an algebraic function:

.. code-block:: python

    from surfaces.test_functions.algebraic._base_algebraic_function import AlgebraicFunction

    class NewFunction(AlgebraicFunction):
        """Description of the function."""

        _spec = {
            "n_dim": 2,
            "default_bounds": (-5.0, 5.0),
            "convex": True,
        }

        f_global = 0.0
        x_global = [0.0, 0.0]

        def _objective(self, params):
            x = params["x0"]
            y = params["x1"]
            return x**2 + y**2

Code of Conduct
===============

Please be respectful and constructive in all interactions. We're building
a welcoming community for everyone interested in optimization.

.. seealso::

   :ref:`developer_template_method`
      Full architecture reference for the template method pattern used
      throughout the class hierarchy.

.. toctree::
   :maxdepth: 1
   :hidden:

   get_involved/contributing
   get_involved/code_of_conduct
