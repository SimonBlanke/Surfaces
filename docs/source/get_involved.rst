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

To add a new test function:

1. Create a new file in the appropriate category:

   - ``test_functions/mathematical/test_functions_1d/`` for 1D functions
   - ``test_functions/mathematical/test_functions_2d/`` for 2D functions
   - ``test_functions/mathematical/test_functions_nd/`` for N-D functions

2. Inherit from the appropriate base class
3. Implement required methods
4. Add to ``__init__.py`` exports
5. Write tests
6. Add documentation

Example structure:

.. code-block:: python

    from .._base_mathematical_function import MathematicalFunction

    class NewFunction(MathematicalFunction):
        """Description of the function.

        Parameters
        ----------
        metric : str, default="loss"
            Either "loss" (minimize) or "score" (maximize).
        sleep : float, default=0
            Artificial delay per evaluation.
        """

        default_bounds = (-5.0, 5.0)

        def __init__(self, metric="loss", sleep=0):
            super().__init__(metric=metric, sleep=sleep)

        def _create_objective_function(self):
            def objective(params):
                x = params["x0"]
                y = params["x1"]
                return x**2 + y**2  # Example
            self.pure_objective_function = objective

Code of Conduct
===============

Please be respectful and constructive in all interactions. We're building
a welcoming community for everyone interested in optimization.

.. toctree::
   :maxdepth: 1
   :hidden:

   get_involved/contributing
   get_involved/code_of_conduct
