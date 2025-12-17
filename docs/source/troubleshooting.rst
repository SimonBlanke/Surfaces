.. _troubleshooting:

===============
Troubleshooting
===============

Solutions to common problems when using Surfaces.

Installation Issues
===================

ModuleNotFoundError: No module named 'surfaces'
-----------------------------------------------

**Problem**: Python can't find the surfaces module.

**Solutions**:

1. Make sure Surfaces is installed:

   .. code-block:: bash

       pip install surfaces

2. Check you're using the right Python environment:

   .. code-block:: bash

       which python
       pip list | grep surfaces

3. If using a virtual environment, make sure it's activated.

Dependency conflicts
--------------------

**Problem**: Version conflicts with other packages.

**Solution**: Create a fresh virtual environment:

.. code-block:: bash

    python -m venv surfaces-env
    source surfaces-env/bin/activate  # On Windows: surfaces-env\Scripts\activate
    pip install surfaces

Runtime Errors
==============

ValueError: Missing required parameters
---------------------------------------

**Problem**: You didn't provide all required parameters.

**Solution**: Check the expected parameters:

.. code-block:: python

    func = SphereFunction(n_dim=3)
    space = func.search_space()
    print(space.keys())  # dict_keys(['x0', 'x1', 'x2'])

    # Make sure to provide all parameters
    result = func({"x0": 1.0, "x1": 2.0, "x2": 3.0})

ValueError: Unexpected parameters
---------------------------------

**Problem**: You provided parameters that don't exist.

**Solution**: Check parameter names match exactly:

.. code-block:: python

    # Wrong: using 'x' instead of 'x0'
    result = func({"x": 1.0, "y": 2.0})  # Raises error

    # Correct: use the exact parameter names
    result = func({"x0": 1.0, "x1": 2.0})

TypeError: evaluate() takes N positional arguments
--------------------------------------------------

**Problem**: Wrong number of arguments to ``evaluate()``.

**Solution**: Provide one value per dimension:

.. code-block:: python

    func = SphereFunction(n_dim=3)

    # Wrong: only 2 values for 3D function
    result = func.evaluate(1.0, 2.0)  # Raises error

    # Correct: provide all 3 values
    result = func.evaluate(1.0, 2.0, 3.0)

Performance Issues
==================

Evaluations are slow
--------------------

**Problem**: Function evaluations take too long.

**Solutions**:

1. Disable validation for performance-critical code:

   .. code-block:: python

       func = SphereFunction(n_dim=3, validate=False)

2. For ML functions, the slowness is inherent to model training.
   Consider using smaller datasets or simpler models.

3. Check if you accidentally set a ``sleep`` delay:

   .. code-block:: python

       func = SphereFunction(n_dim=3, sleep=0)  # Explicitly no delay

scipy Integration Issues
========================

Cannot create scipy bounds for non-numeric parameter
----------------------------------------------------

**Problem**: ``to_scipy()`` fails for functions with categorical parameters.

**Solution**: ML functions with categorical hyperparameters can't be
directly converted to scipy format. Use a different optimization library
that supports mixed parameter types.

Results Issues
==============

Optimization finds wrong minimum
--------------------------------

**Problem**: The optimizer doesn't find the known global minimum.

**Solutions**:

1. Check you're using the loss mode (minimization):

   .. code-block:: python

       result = func.loss(params)  # Not func.score(params)

2. The optimizer might be stuck in a local minimum. Try:

   - Different starting points
   - More iterations
   - A global optimization algorithm

3. Verify the search space bounds are appropriate.

Different results each run
--------------------------

**Problem**: Results vary between runs.

**Cause**: The optimization algorithm uses random initialization.

**Solution**: Set a random seed for reproducibility:

.. code-block:: python

    import numpy as np
    np.random.seed(42)

    # Then run your optimization

Getting Help
============

If you can't find a solution:

1. Check the :doc:`faq` for common questions
2. Search `existing issues <https://github.com/SimonBlanke/Surfaces/issues>`_
3. Open a new issue with:

   - Surfaces version (``surfaces.__version__``)
   - Python version
   - Operating system
   - Minimal code to reproduce the problem
   - Full error traceback
