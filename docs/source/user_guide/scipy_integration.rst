.. _user_guide_scipy_integration:

=================
scipy Integration
=================

Surfaces provides built-in support for scipy.optimize, making it easy
to use test functions with scipy's optimization algorithms.

The to_scipy() Method
=====================

Every test function with numeric parameters can be converted to scipy format:

.. code-block:: python

    from surfaces.test_functions import SphereFunction
    from scipy.optimize import minimize

    # Create the test function
    func = SphereFunction(n_dim=3)

    # Convert to scipy format
    objective, bounds, x0 = func.to_scipy()

The method returns:

- ``objective``: A callable that takes a numpy array
- ``bounds``: A ``scipy.optimize.Bounds`` object
- ``x0``: Initial guess (center of bounds)

Basic Example
=============

.. code-block:: python

    from surfaces.test_functions import RosenbrockFunction
    from scipy.optimize import minimize

    # Create a 5D Rosenbrock function
    func = RosenbrockFunction(n_dim=5)

    # Convert to scipy format
    objective, bounds, x0 = func.to_scipy()

    # Run L-BFGS-B optimization
    result = minimize(
        objective,
        x0,
        bounds=bounds,
        method='L-BFGS-B'
    )

    print(f"Success: {result.success}")
    print(f"Optimal x: {result.x}")
    print(f"Minimum value: {result.fun}")

Using Different Optimizers
==========================

The scipy-compatible interface works with any scipy optimizer that
accepts bounds:

L-BFGS-B
--------

.. code-block:: python

    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

SLSQP
-----

.. code-block:: python

    result = minimize(objective, x0, bounds=bounds, method='SLSQP')

Powell
------

.. code-block:: python

    result = minimize(objective, x0, method='Powell')

Trust-Region Methods
--------------------

.. code-block:: python

    result = minimize(
        objective,
        x0,
        bounds=bounds,
        method='trust-constr'
    )

Global Optimization
===================

scipy also provides global optimization methods:

Differential Evolution
----------------------

.. code-block:: python

    from scipy.optimize import differential_evolution

    func = RastriginFunction(n_dim=5)
    objective, bounds, x0 = func.to_scipy()

    # Convert Bounds to list of tuples for differential_evolution
    lower, upper = func.get_bounds()
    bounds_list = list(zip(lower, upper))

    result = differential_evolution(objective, bounds_list)

Basin Hopping
-------------

.. code-block:: python

    from scipy.optimize import basinhopping

    func = AckleyFunction()
    objective, bounds, x0 = func.to_scipy()

    result = basinhopping(objective, x0, minimizer_kwargs={'bounds': bounds})

Dual Annealing
--------------

.. code-block:: python

    from scipy.optimize import dual_annealing

    func = RastriginFunction(n_dim=3)
    objective, bounds, x0 = func.to_scipy()
    lower, upper = func.get_bounds()
    bounds_list = list(zip(lower, upper))

    result = dual_annealing(objective, bounds_list)

Working with Bounds
===================

The ``get_bounds()`` method returns bounds as numpy arrays:

.. code-block:: python

    func = SphereFunction(n_dim=3)

    lower, upper = func.get_bounds()
    print(f"Lower bounds: {lower}")
    print(f"Upper bounds: {upper}")

    # Convert to different formats
    bounds_list = list(zip(lower, upper))  # [(low0, high0), ...]

Custom Initial Points
=====================

The default x0 is the center of bounds, but you can use any starting point:

.. code-block:: python

    import numpy as np

    func = SphereFunction(n_dim=3)
    objective, bounds, x0 = func.to_scipy()

    # Use a random starting point
    lower, upper = func.get_bounds()
    x0_random = np.random.uniform(lower, upper)

    result = minimize(objective, x0_random, bounds=bounds, method='L-BFGS-B')

Limitations
===========

The ``to_scipy()`` method has some limitations:

1. **Numeric parameters only**: Functions with categorical parameters
   (like ML functions) cannot be converted.

2. **Continuous optimization**: scipy treats the search space as continuous,
   even if the original function uses discrete values.

3. **No parallelization**: scipy's optimizers are single-threaded.

For functions with categorical parameters, use optimization libraries
that support mixed parameter types (Hyperactive, Optuna, etc.).

Performance Tips
================

1. **Disable validation**: The scipy objective function bypasses
   validation for performance.

2. **Use appropriate tolerances**: Adjust ``tol`` and ``options``
   for your needs:

   .. code-block:: python

       result = minimize(
           objective,
           x0,
           bounds=bounds,
           method='L-BFGS-B',
           options={'maxiter': 1000, 'gtol': 1e-8}
       )

3. **Consider global methods**: For multimodal functions, local
   optimizers may get stuck in local minima.
