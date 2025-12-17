.. _contributing:

============
Contributing
============

Thank you for your interest in contributing to Surfaces!

Getting Started
===============

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up a development environment
4. Create a feature branch
5. Make your changes
6. Submit a pull request

Development Setup
=================

.. code-block:: bash

    # Clone your fork
    git clone https://github.com/YOUR_USERNAME/Surfaces.git
    cd Surfaces

    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install in editable mode with dev dependencies
    pip install -e ".[dev,test]"

Running Tests
=============

.. code-block:: bash

    # Run all tests
    make test

    # Run pytest only
    make py-test

    # Run linting
    flake8

Code Style
==========

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Include type hints where appropriate

Adding Test Functions
=====================

To add a new test function:

1. Create a new file in the appropriate directory
2. Inherit from the correct base class
3. Implement required methods
4. Add to ``__init__.py`` exports
5. Write tests
6. Update documentation

Submitting Changes
==================

1. Ensure all tests pass
2. Update documentation if needed
3. Write a clear commit message
4. Push to your fork
5. Open a pull request

Pull Request Guidelines
=======================

- Keep changes focused and atomic
- Include tests for new functionality
- Update documentation as needed
- Reference any related issues

Questions?
==========

Open an issue on GitHub if you have questions about contributing.
